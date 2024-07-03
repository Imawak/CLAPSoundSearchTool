import sys
import numpy as np
import librosa
import os
import csv
import time
import faiss
from sklearn.preprocessing import normalize
import math
import laion_clap
import clap_module
import re
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QMessageBox, QTableView, QStyledItemDelegate
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QColor, QDesktopServices,QMouseEvent, QDrag, QDragEnterEvent, QDropEvent, QIcon, QPixmap, QPainter
from PyQt6.QtCore import Qt, QEvent, QUrl, QThread, pyqtSignal, QMimeData

from Ui_Main import Ui_MainWindow
from Ui_SoundInfoGeneration import Ui_Form_SoundInfoGeneration
from Ui_TextSearchSound import Ui_Form_TextSearchSound
from Ui_SoundSearchSound import Ui_Form_SoundSearchSound

# quantization
# 定义量化函数，用于将音频数据在整数形式（int16）和浮点数形式（float32）之间进行转换
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

if not os.path.exists('./CLAPSoundSearchTool/Data/'):
    os.makedirs('./CLAPSoundSearchTool/Data/')

# 音频文件所在文件夹路径
default_audio_folder_path = './CLAPSoundSearchTool/Data/Audio'

# csv 文件生成路径
default_csv_file_path = './CLAPSoundSearchTool/Data/audio_embeddings.csv'

# parquet 文件生成路径
default_parquet_file_path = './CLAPSoundSearchTool/Data/audio_embeddings.parquet'

# ckpt 文件路径
default_ckpt_file_path = './CLAPSoundSearchTool/Data/ckpt_file/630k-audioset-best.pt'

# 默认路径txt文件路径
default_path_file = './CLAPSoundSearchTool/Data/DefaultPath.txt'

# 路径中斜杠自动替换
def normalize_path(path):
    return path.replace('\\', '/')

# 检查 DefaultPath.txt 是否存在
if os.path.exists(default_path_file):
    # 读取文件中的路径
    with open(default_path_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if len(lines) >= 4:
            default_audio_folder_path = normalize_path(os.path.abspath(lines[0].strip()))
            default_csv_file_path = normalize_path(os.path.abspath(lines[1].strip()))
            default_parquet_file_path = normalize_path(os.path.abspath(lines[2].strip()))
            default_ckpt_file_path = normalize_path(os.path.abspath(lines[3].strip()))
else:
    # 创建文件并写入默认路径
    with open(default_path_file, 'w', encoding='utf-8') as file:
        file.write(f"{normalize_path(default_audio_folder_path)}\n")
        file.write(f"{normalize_path(default_csv_file_path)}\n")
        file.write(f"{normalize_path(default_parquet_file_path)}\n")
        file.write(f"{normalize_path(default_ckpt_file_path)}\n")

# 检查 音频 文件路径是否存在，如果不存在则创建
if not os.path.exists(default_audio_folder_path):
    os.makedirs(default_audio_folder_path)

# 检查 ckpt 文件夹路径是否存在，如果不存在则创建
if not os.path.exists('./CLAPSoundSearchTool/Data/ckpt_file'):
    os.makedirs('./CLAPSoundSearchTool/Data/ckpt_file')

# 初始化文件列表
files = []

# 创建 CLAP 模型对象并加载预训练模型
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt(ckpt = default_ckpt_file_path)


## 主界面窗体
class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent=parent)
        self.setupUi(self)
        self.pushButton_SoundInfoGeneration.clicked.connect(self.OpenSoundInfoGeneration)
        self.pushButton_TextSearchSound.clicked.connect(self.OpenTextSearchSound)
        self.pushButton_SoundSearchSound.clicked.connect(self.OpenSoundSearchSound)
        self.setWindowIcon(QIcon('./ICON.ico'))

    def OpenSoundInfoGeneration(self):
        self.SoundInfoGenerationWindow = SoundInfoGenerationWindow()
        self.SoundInfoGenerationWindow.show()

    def OpenTextSearchSound(self):
        self.TextSearchSound = TextSearchSoundWindow()
        self.TextSearchSound.show()

    def OpenSoundSearchSound(self):
        self.SoundSearchSound = SoundSearchSoundWindow()
        self.SoundSearchSound.show()



## 超链接
class HyperlinkDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super(HyperlinkDelegate, self).__init__(parent)

    def createEditor(self, parent, option, index):
        return None

    def paint(self, painter, option, index):
        text = index.data()
        if isinstance(text, str):
            painter.setPen(Qt.GlobalColor.blue)
            painter.drawText(option.rect, int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter), text)

    # 点击打开音频文件/拖拽移动音频文件
    def editorEvent(self, event, model, option, index):
        if isinstance(event, QMouseEvent):
            if event.type() == QMouseEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self.startPos = event.position().toPoint()
            elif event.type() == QMouseEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                text = index.data()
                if isinstance(text, str):
                    QDesktopServices.openUrl(QUrl.fromLocalFile(text))
                return True
            elif event.type() == QMouseEvent.Type.MouseMove and (event.buttons() & Qt.MouseButton.LeftButton):
                if (event.position().toPoint() - self.startPos).manhattanLength() > QApplication.startDragDistance():
                    text = index.data()
                    if isinstance(text, str) and text.endswith('.wav'):
                        # 获取视图对象
                        view = option.widget
                        drag = QDrag(view)
                        mimeData = QMimeData()
                        mimeData.setUrls([QUrl.fromLocalFile(text)])
                        drag.setMimeData(mimeData)
                        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)
                        return True
        return False


## Parquet 文件加载线程
class LoadParquetWorker(QThread):
    progress_update = pyqtSignal(int)
    message_update = pyqtSignal(str)
    data_loaded = pyqtSignal(object)

    def __init__(self, parquet_file_path):
        super().__init__()
        self.parquet_file_path = parquet_file_path

    def run(self):
        self.message_update.emit('加载Parquet文件中，请稍等')
        
        # 使用pandas读取Parquet文件
        df = pd.read_parquet(self.parquet_file_path)
        
        # 打印列名以进行调试
        print("Parquet 文件的列名：", df.columns.tolist())
        
        if '音频嵌入向量' not in df.columns or '文件名嵌入向量' not in df.columns:
            raise KeyError("Parquet 文件中缺少 '音频嵌入向量' 或 '文件名嵌入向量' 列")

        # 使用矢量化操作转换嵌入向量字符串为NumPy数组
        def vectorized_convert(embedding_str_series):
            return np.vstack(embedding_str_series.str[1:-1].str.split().apply(lambda x: np.array(x, dtype=np.float32)).values)

        audio_embeddings = vectorized_convert(df['音频嵌入向量'])
        filename_embeddings = vectorized_convert(df['文件名嵌入向量'])
        filenames = df['文件名'].apply(lambda x: os.path.splitext(x)[0].lower()).tolist()
        
        self.data_loaded.emit((audio_embeddings, filename_embeddings, filenames, df))
        self.message_update.emit('Parquet文件加载完成')


## 文本搜索音频线程
class TextSearchSoundWorker(QThread):
    progress_update = pyqtSignal(int)
    message_update = pyqtSignal(str)
    result_update = pyqtSignal(object)

    def __init__(self, text, k, audio_embeddings, filename_embeddings, filenames, df, combine_weight):
        super().__init__()
        self.text = text
        self.k = k
        self.audio_embeddings = audio_embeddings
        self.filename_embeddings = filename_embeddings
        self.filenames = filenames
        self.df = df
        self.combine_weight = combine_weight

    def run(self):
        start_time = time.time()

        self.message_update.emit('查询中，请稍等')

        text_data = ['']
        text_data.append(self.text)
        text_embed = model.get_text_embedding(text_data)

        embed_time = time.time()

        # 若音频数量不足k则检索全部
        Parquet_count = len(self.df)
        if self.k <= Parquet_count:
            k = self.k
        else:
            k = Parquet_count

        audio_embeddings = normalize(self.audio_embeddings, axis=1, norm='l2')
        filename_embeddings = normalize(self.filename_embeddings, axis=1, norm='l2')

        embed_normalize_time = time.time()

        jaccard_start_time = time.time()

        def jaccard_similarity(s1, s2):
            s1_tokens = set(re.split(r'[ _]', s1.lower()))
            s2_tokens = set(re.split(r'[ _]', s2.lower()))
            intersection = len(s1_tokens.intersection(s2_tokens))
            total_words = len(s1_tokens) + len(s2_tokens) - intersection
            return intersection / total_words if total_words else 0

        jaccard_count = 0
        weighted_filename_embeds = np.zeros_like(filename_embeddings)

        keywords = ['short', 'medium', 'long', 'heavy', 'soft', 'quick', 'fast', 'slow', 'large', 'small', 'distant', 'close', 'high', 'low', 'mid']
        text_tokens = set(re.split(r'[ _]', self.text.lower()))

        results = []

        for i, filename in enumerate(self.filenames):
            jaccard_sim = jaccard_similarity(self.text, filename)
            filename_tokens = set(re.split(r'[ _]', filename.lower()))
            common_keywords = filename_tokens.intersection(text_tokens)
            keywordweight = 1.5 if any(keyword in common_keywords for keyword in keywords) else 1.0

            weighted_filename_embeds[i] = (1 - ((1 - jaccard_sim) * 0.75)) * filename_embeddings[i] * keywordweight
            combined_embedding = (1 - self.combine_weight / 100) * audio_embeddings[i] + (self.combine_weight / 100) * weighted_filename_embeds[i]
            results.append([jaccard_sim, weighted_filename_embeds[i].tolist(), combined_embedding.tolist()])
            jaccard_count += 1

        result_df = pd.DataFrame(results, columns=['jaccard_similarity', 'weighted_filename_embed', 'combined_embedding'])

        jaccard_end_time = time.time()

        dimension = len(results[0][2])
        index = faiss.IndexFlatIP(dimension)
        
        for combined_embedding in result_df['combined_embedding']:
            index.add(np.array([combined_embedding]))

        faiss_index_time = time.time()

        Qmodel = QStandardItemModel()
        Qmodel.setHorizontalHeaderLabels(['文件路径', '文件名', '相似度'])

        for i, text_vec in enumerate(text_embed[1:], start=1):
            text_vec = normalize(text_vec.reshape(1, -1), axis=1, norm='l2')

            D, I = index.search(text_vec, k)

            similarities = []
            for dist, idx in zip(D[0], I[0]):
                similarity = dist
                if similarity > 1:
                    similarity = 1
                elif similarity < -1:
                    similarity = -1
                file_path = self.df.loc[idx, '文件路径']
                filename = self.df.loc[idx, '文件名']
                similarity_percentage = math.acos(-similarity) / math.pi * 100
                similarities.append((file_path, filename, similarity_percentage))

            similarities.sort(key=lambda x: x[2], reverse=True)

            faiss_end_time = time.time()

            for file_path, filename, similarity_percentage in similarities:
                row = [
                    QStandardItem(file_path),
                    QStandardItem(filename),
                    QStandardItem(f"{similarity_percentage:.4f}%")
                ]
                Qmodel.appendRow(row)

        end_time = time.time()

        print(f"文本嵌入向量获取时间: {embed_time - start_time:.2f}秒")
        print(f"嵌入向量归一化时间: {embed_normalize_time - embed_time:.2f}秒")
        print(f"Jaccard相似度计算时间: {jaccard_end_time - jaccard_start_time:.2f}秒")
        print(f"创建并填充FAISS索引时间: {faiss_index_time - jaccard_end_time:.2f}秒")
        print(f"FAISS检索并排序时间: {faiss_end_time - faiss_index_time:.2f}秒")
        print(f"总检索时间: {end_time - start_time:.2f}秒")

        self.message_update.emit(f"Jaccard耗时: {jaccard_end_time - jaccard_start_time:.2f}秒\n" + f"检索耗时: {faiss_end_time - faiss_index_time:.2f}秒\n" + f"总耗时: {end_time - start_time:.2f}秒\n")
        self.result_update.emit(Qmodel)


## 文本搜索音频窗体
class TextSearchSoundWindow(QWidget, Ui_Form_TextSearchSound):
    def __init__(self, parent=None):
        super(TextSearchSoundWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.pushButton_TextSearchSound.clicked.connect(self.TextSearchSoundStart)
        self.pushButton_LoadParquet.clicked.connect(self.loadParquet)
        self.setWindowIcon(QIcon('./ICON.ico'))
        self.tableView_TextSearchSound.setItemDelegateForColumn(0, HyperlinkDelegate(self))
        self.lineEdit_ParquetFilePath.setAcceptDrops(True)
        self.lineEdit_ParquetFilePath.dragEnterEvent = self.dragEnterEvent
        self.lineEdit_ParquetFilePath.dropEvent = self.parquet_dropEvent

        self.audio_embeddings = None
        self.filename_embeddings = None
        self.filenames = None
        self.df = None

    def loadParquet(self):
        parquet_file_path_input = self.lineEdit_ParquetFilePath.text()
        parquet_file_path = parquet_file_path_input.replace('\\', '/') if parquet_file_path_input else default_parquet_file_path

        self.parquet_worker = LoadParquetWorker(parquet_file_path)
        self.parquet_worker.message_update.connect(self.showMessage)
        self.parquet_worker.data_loaded.connect(self.setParquetData)
        self.parquet_worker.start()

    def setParquetData(self, data):
        self.audio_embeddings, self.filename_embeddings, self.filenames, self.df = data

    def TextSearchSoundStart(self):
        text = self.lineEdit_TextSearchSound.text()
        if text and self.audio_embeddings is not None:
            k = int(self.spinBox_TextSearchSound.text())
            combine_weight = float(self.spinBox_CombineWeight.text())

            self.worker = TextSearchSoundWorker(text, k, self.audio_embeddings, self.filename_embeddings, self.filenames, self.df, combine_weight)
            self.worker.message_update.connect(self.showMessage)
            self.worker.result_update.connect(self.updateResults)
            self.worker.start()
        else:
            QMessageBox.information(self, '提示', '请输入文本并加载Parquet文件后再进行搜索')

    def showMessage(self, message):
        QMessageBox.information(self, '消息', message)

    def updateResults(self, Qmodel):
        self.tableView_TextSearchSound.setModel(Qmodel)
        self.tableView_TextSearchSound.setColumnWidth(0, 1000)
        self.tableView_TextSearchSound.setColumnWidth(1, 400)
        self.tableView_TextSearchSound.setColumnWidth(2, 100)
        self.tableView_TextSearchSound.show()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def parquet_dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if len(urls) > 0:
            file_path = urls[0].toLocalFile()
            self.lineEdit_ParquetFilePath.setText(file_path)


## 音频搜索音频线程
class SoundSearchSoundWorker(QThread):
    message_update = pyqtSignal(str)
    result_ready = pyqtSignal(list)

    def __init__(self, audio_file_path, audio_embeddings, filename_embeddings, filenames, df, k):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.audio_embeddings = audio_embeddings
        self.filename_embeddings = filename_embeddings
        self.filenames = filenames
        self.df = df
        self.k = k

    def run(self):
        start_time = time.time()

        self.message_update.emit('查询中，请稍等')

        # 从音频数据获取音频嵌入向量的函数
        def get_audio_embedding(audio_file_path):
            audio_data, _ = librosa.load(audio_file_path, sr=None)
            audio_data = audio_data.reshape(1, -1)  # 转换为 (1, T) 形状
            audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
            return audio_embed

        # 获取输入音频嵌入向量
        input_start_time = time.time()
        input_audio_embed = get_audio_embedding(self.audio_file_path)
        input_end_time = time.time()

        processing_start_time = time.time()

        Parquet_count = len(self.audio_embeddings)
        if self.k > Parquet_count:
            self.k = Parquet_count

        # 转换音频嵌入向量为numpy数组，并进行归一化
        audio_embeddings = normalize(self.audio_embeddings, axis=1, norm='l2')
        processing_end_time = time.time()

        walk_time = time.time()

        # 创建FAISS索引
        faiss_start_time = time.time()
        dimension = audio_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(audio_embeddings)

        # 对输入音频嵌入向量进行归一化
        input_audio_embed = normalize(input_audio_embed.reshape(1, -1), axis=1, norm='l2')

        # 使用FAISS进行检索
        D, I = index.search(input_audio_embed, self.k)
        faiss_end_time = time.time()

        similarities = []
        for dist, idx in zip(D[0], I[0]):
            similarity = dist
            file_path, filename, audio_embed = self.df.iloc[idx][['文件路径', '文件名', '音频嵌入向量']]
            similarity_percentage = math.acos(-similarity) / math.pi * 100
            similarities.append((file_path, filename, similarity_percentage))

        similarities.sort(key=lambda x: x[2], reverse=True)

        end_time = time.time()

        print(
            f'输入音频处理时间: {input_end_time - input_start_time:.2f}秒\n' +
            f'数据处理时间: {processing_end_time - processing_start_time:.2f}秒\n' +
            f'FAISS索引创建和检索时间: {faiss_end_time - faiss_start_time:.2f}秒\n' +
            f'总耗时: {end_time - start_time:.2f}秒\n'
        )

        self.result_ready.emit(similarities)
        self.message_update.emit(f'检索耗时: {end_time - walk_time:.2f}秒\n' + f'总耗时: {end_time - start_time:.2f}秒')


## 音频搜索音频窗体
class SoundSearchSoundWindow(QWidget, Ui_Form_SoundSearchSound):
    def __init__(self, parent=None):
        super(SoundSearchSoundWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.pushButton_SoundSearchSound.clicked.connect(self.SoundSearchSoundStart)
        self.pushButton_LoadParquet.clicked.connect(self.loadParquet)  # 新增按钮
        self.setWindowIcon(QIcon('./ICON.ico'))
        self.tableView_SoundSearchSound.setItemDelegateForColumn(0, HyperlinkDelegate(self))
        self.lineEdit_ParquetFilePath.setAcceptDrops(True)
        self.lineEdit_ParquetFilePath.dragEnterEvent = self.dragEnterEvent
        self.lineEdit_ParquetFilePath.dropEvent = self.parquet_dropEvent
        self.lineEdit_InputSoundPath.setAcceptDrops(True)
        self.lineEdit_InputSoundPath.dragEnterEvent = self.dragEnterEvent
        self.lineEdit_InputSoundPath.dropEvent = self.sound_dropEvent

        self.audio_embeddings = None
        self.filename_embeddings = None
        self.filenames = None
        self.df = None

    def loadParquet(self):
        parquet_file_path_input = self.lineEdit_ParquetFilePath.text()
        parquet_file_path = parquet_file_path_input.replace('\\', '/') if parquet_file_path_input else default_parquet_file_path

        self.parquet_worker = LoadParquetWorker(parquet_file_path)
        self.parquet_worker.message_update.connect(self.show_message)
        self.parquet_worker.data_loaded.connect(self.setParquetData)
        self.parquet_worker.start()

    def setParquetData(self, data):
        self.audio_embeddings, self.filename_embeddings, self.filenames, self.df = data

    def SoundSearchSoundStart(self):
        audio_file_path = self.lineEdit_InputSoundPath.text()
        if audio_file_path and self.audio_embeddings is not None:
            k = int(self.spinBox_SoundSearchSound.text())

            self.worker = SoundSearchSoundWorker(audio_file_path, self.audio_embeddings, self.filename_embeddings, self.filenames, self.df, k)
            self.worker.message_update.connect(self.show_message)
            self.worker.result_ready.connect(self.display_results)
            self.worker.start()
        else:
            QMessageBox.information(self, '提示', '请输入音频文件并加载Parquet文件后再进行搜索')

    def show_message(self, message):
        QMessageBox.information(self, '消息', message)

    def display_results(self, similarities):
        Qmodel = QStandardItemModel()
        Qmodel.setHorizontalHeaderLabels(['文件路径', '文件名', '相似度'])

        for file_path, filename, similarity_percentage in similarities:
            row = [
                QStandardItem(file_path),
                QStandardItem(filename),
                QStandardItem(f"{similarity_percentage:.4f}%")
            ]
            Qmodel.appendRow(row)

        self.tableView_SoundSearchSound.setModel(Qmodel)
        self.tableView_SoundSearchSound.setColumnWidth(0, 1000)
        self.tableView_SoundSearchSound.setColumnWidth(1, 400)
        self.tableView_SoundSearchSound.setColumnWidth(2, 100)
        self.tableView_SoundSearchSound.show()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def sound_dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if len(urls) > 0:
            file_path = urls[0].toLocalFile()
            self.lineEdit_InputSoundPath.setText(file_path)

    def parquet_dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if len(urls) > 0:
            file_path = urls[0].toLocalFile()
            self.lineEdit_ParquetFilePath.setText(file_path)


## 音频数据初始化线程
class SoundInfoGenerationWorker(QThread):
    progress_update = pyqtSignal(int)
    message_update = pyqtSignal(str)

    def __init__(self, audio_folder_path, csv_file_path):
        super().__init__()
        self.audio_folder_path = audio_folder_path
        self.csv_file_path = csv_file_path

    def run(self):
        files = []

        # 如果音频文件夹路径不存在，则创建该路径
        if not os.path.exists(self.audio_folder_path):
            os.makedirs(self.audio_folder_path)
            self.message_update.emit(f'已自动创建音频文件夹：'+self.audio_folder_path)

        # 第一次遍历文件夹及其子文件夹，获取所有 .wav 文件路径
        for root, dirs, filenames in os.walk(self.audio_folder_path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    file_path = os.path.join(root, filename)
                    files.append(file_path)

        # wav文件总数量
        total_files = len(files)
        self.message_update.emit('搜索到的WAV文件总数：' + str(total_files))

        # 统计开始时间
        start_time = time.time()

        # 创建 CLAP 模型对象并加载预训练模型
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt(ckpt=default_ckpt_file_path)

        # 从音频数据获取音频嵌入向量的函数
        def get_audio_embedding(audio_data):
            audio_data = audio_data.reshape(1, -1)  # 转换为 (1,T) 或 (N,T) 的形状
            audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
            return audio_embed

        # 获取文件名嵌入向量的函数
        def get_filename_embedding(filename):
            filenames = ['']  # 填充一个空字符串
            filenames.append(filename)
            filename_embeds = model.get_text_embedding(filenames)
            filename_embed = filename_embeds[1]  # 获取索引为1的嵌入向量
            filename_embed = normalize(filename_embed.reshape(1, -1), axis=1, norm='l2')
            return filename_embed
        
        # 初始化 CSV 文件并写入表头
        with open(self.csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['索引', '文件路径', '文件名', '音频嵌入向量', '文件名嵌入向量'])

        # 初始化索引计数器
        idx_counter = 0

        # 第二次遍历文件列表，提取嵌入向量并保存数据
        for file_path in files:
            audio_data, _ = librosa.load(file_path, sr=48000)
            audio_embed = get_audio_embedding(audio_data)
            filename = os.path.basename(file_path)
            filename_clean = os.path.splitext(filename)[0].lower()

            filename_embed = get_filename_embedding(filename_clean)[0]

            with open(self.csv_file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([idx_counter, file_path, filename_clean, '[' + ' '.join(map(str, audio_embed.flatten())) + ']', '[' + ' '.join(map(str, filename_embed)) + ']'])

            idx_counter += 1
            self.progress_update.emit(idx_counter + 1)
            print(idx_counter)

        self.message_update.emit('已创建包含音频嵌入向量的 CSV 文件:' + self.csv_file_path)

        # csv同目录下生成parquet文件
        df = pd.read_csv(self.csv_file_path)
        table = pa.Table.from_pandas(df)
        csv_dir = os.path.dirname(self.csv_file_path)
        csv_filename = os.path.basename(self.csv_file_path)
        # 将文件扩展名从.csv更改为.parquet
        parquet_filename = os.path.splitext(csv_filename)[0] + '.parquet'
        # 生成Parquet文件路径
        parquet_file_path = os.path.join(csv_dir, parquet_filename)
        pq.write_table(table, parquet_file_path)

        self.message_update.emit('已创建包含音频嵌入向量的 Parquet 文件:' + parquet_file_path)

        # 统计结束时间
        end_time = time.time()

        # 显示运行时长
        self.message_update.emit("总耗时: {:.2f}秒".format(end_time - start_time))


## 音频数据刷新线程
class SoundInfoRefreshWorker(QThread):
    progress_update = pyqtSignal(int)
    message_update = pyqtSignal(str)

    def __init__(self, audio_folder_path, csv_file_path):
        super().__init__()
        self.audio_folder_path = audio_folder_path
        self.csv_file_path = csv_file_path

    def run(self):
        files = []

        if not os.path.exists(self.audio_folder_path):
            os.makedirs(self.audio_folder_path)
            self.message_update.emit('已自动创建音频文件夹：' + self.audio_folder_path)

        current_files = set()
        for root, dirs, filenames in os.walk(self.audio_folder_path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    file_path = os.path.join(root, filename)
                    current_files.add(file_path)

        existing_data = []
        if os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader)
                for row in reader:
                    existing_data.append(row)

        existing_file_paths = {row[1]: row[4] for row in existing_data}

        files_to_add = current_files - set(existing_file_paths.keys())
        files_to_remove = set(existing_file_paths.keys()) - current_files

        num_files_to_add = len(files_to_add)
        num_files_to_remove = len(files_to_remove)

        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt(ckpt=default_ckpt_file_path)

        def get_audio_embedding(audio_data):
            audio_data = audio_data.reshape(1, -1)
            audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
            return audio_embed

        def get_filename_embedding(filename):
            filenames = ['']
            filenames.append(filename)
            filename_embeds = model.get_text_embedding(filenames)
            filename_embed = filename_embeds[1]
            filename_embed = normalize(filename_embed.reshape(1, -1), axis=1, norm='l2')
            return filename_embed

        new_data = []
        add_counter = 0
        for file_path in files_to_add:
            audio_data, _ = librosa.load(file_path, sr=48000)
            audio_embed = get_audio_embedding(audio_data)
            filename = os.path.basename(file_path)
            filename_clean = os.path.splitext(filename)[0].lower()

            filename_embed = get_filename_embedding(filename_clean)[0]

            new_data.append([None, file_path, filename_clean, audio_embed, filename_embed])
            add_counter += 1
            self.progress_update.emit(add_counter)

        updated_data = [row for row in existing_data if row[1] not in files_to_remove]
        updated_data.extend(new_data)

        for idx, row in enumerate(updated_data):
            row[0] = idx

        with open(self.csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['索引', '文件路径', '文件名', '音频嵌入向量', '文件名嵌入向量'])
            for row in updated_data:
                writer.writerow([row[0], row[1], row[2], row[3].flatten() if isinstance(row[3], np.ndarray) else row[3], row[4]])

        self.message_update.emit(f'新增文件数：{num_files_to_add}\n删除文件数：{num_files_to_remove}')
        self.message_update.emit('已更新包含音频嵌入向量的 CSV 文件:' + self.csv_file_path)
        
        # csv同目录下生成parquet文件
        df = pd.read_csv(self.csv_file_path)
        table = pa.Table.from_pandas(df)
        csv_dir = os.path.dirname(self.csv_file_path)
        csv_filename = os.path.basename(self.csv_file_path)
        # 将文件扩展名从.csv更改为.parquet
        parquet_filename = os.path.splitext(csv_filename)[0] + '.parquet'
        # 生成Parquet文件路径
        parquet_file_path = os.path.join(csv_dir, parquet_filename)
        pq.write_table(table, parquet_file_path)

        self.message_update.emit('已创建包含音频嵌入向量的 Parquet 文件:' + parquet_file_path)


## 音频数据初始化/刷新窗体
class SoundInfoGenerationWindow(QWidget,Ui_Form_SoundInfoGeneration):
    def __init__(self):
        super(SoundInfoGenerationWindow,self).__init__()
        self.setupUi(self)
        self.pushButton_SoundInfoGeneration.clicked.connect(self.GenerationStart) 
        self.pushButton_SoundInfoRefresh.clicked.connect(self.RefreshStart) 

    def showMessage(self, message):
        QMessageBox.information(self, '消息', message)

    # 音频数据初始化操作确认
    def GenerationStart(self):
        a = QMessageBox.question(self, '初始化', '您确定要进行音频数据初始化吗?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if a == QMessageBox.StandardButton.Yes: 
            self.GenerationProgress()
        else:
            QMessageBox.information(self,'取消初始化','已取消音频数据初始化')

    # 音频数据初始化
    def GenerationProgress(self):
        # 从 QLineEdit 获取音频文件夹路径和 CSV 文件路径
        audio_folder_path_input = self.lineEdit_GenerationInputPath.text()
        csv_file_path_input = self.lineEdit_GenerationOutputPath.text()

        # 如果输入不为空，则使用输入的路径，否则使用默认路径
        audio_folder_path = audio_folder_path_input.replace('\\','/') if audio_folder_path_input else default_audio_folder_path
        csv_file_path = csv_file_path_input.replace('\\','/') if csv_file_path_input else default_csv_file_path

        # 计算文件总数以设置进度条最大值
        files = []
        for root, dirs, filenames in os.walk(audio_folder_path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    files.append(os.path.join(root, filename))
        
        total_files = len(files)
        self.progressBar.reset()
        self.progressBar.setMaximum(total_files)

        self.worker = SoundInfoGenerationWorker(audio_folder_path, csv_file_path)
        self.worker.progress_update.connect(self.progressBar.setValue)
        self.worker.message_update.connect(self.showMessage)
        self.worker.start()
        

    # 音频数据刷新操作确认
    def RefreshStart(self):
        b = QMessageBox.question(self, '刷新', '您确定要进行音频数据刷新吗?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if b == QMessageBox.StandardButton.Yes: 
            self.RefreshProgress()
        else:
            QMessageBox.information(self,'取消刷新','已取消音频数据刷新')      
    
    # 音频数据刷新
    def RefreshProgress(self):
        # 从 QLineEdit 获取音频文件夹路径和 CSV 文件路径
        audio_folder_path_input = self.lineEdit_GenerationInputPath.text()
        csv_file_path_input = self.lineEdit_GenerationOutputPath.text()

        # 如果输入不为空，则使用输入的路径，否则使用默认路径
        audio_folder_path = audio_folder_path_input.replace('\\','/') if audio_folder_path_input else default_audio_folder_path
        csv_file_path = csv_file_path_input.replace('\\','/') if csv_file_path_input else default_csv_file_path

        # 获取当前文件夹中所有 .wav 文件路径
        current_files = set()
        for root, dirs, filenames in os.walk(audio_folder_path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    file_path = os.path.join(root, filename)
                    current_files.add(file_path)

        # 读取现有 CSV 文件中的数据
        existing_data = []
        if os.path.exists(csv_file_path):
            with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader)  # 跳过标题行
                for row in reader:
                    existing_data.append(row)

        # 提取 CSV 文件中已存在的文件路径
        existing_file_paths = set(row[1] for row in existing_data)

        # 找出需要添加和删除的文件
        files_to_add = current_files - existing_file_paths
        files_to_remove = existing_file_paths - current_files

        self.progressBar.setMaximum(len(files_to_add))

        self.worker = SoundInfoRefreshWorker(audio_folder_path, csv_file_path)
        self.worker.progress_update.connect(self.progressBar.setValue)
        self.worker.message_update.connect(self.showMessage)
        self.worker.start()

## 主函数
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    soundinfogenerationwindow = SoundInfoGenerationWindow()
    mainwindow.show()
    sys.exit(app.exec())