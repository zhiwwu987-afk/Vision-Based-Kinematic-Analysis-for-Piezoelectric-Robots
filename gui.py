# gui_main.py
import os
import sys
import threading
import random

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox, QSizePolicy, QSpacerItem
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QUrl, Signal, Slot

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from trace import main as core_main


class TrackerGUI(QWidget):
    log_signal = Signal(str)
    video_loaded = Signal(str)
    tracking_done = Signal()

    def __init__(self):
        super().__init__()

        self.generated_videos = []
        self.last_video_path = None

        self.media_ended = False

        self.setWindowTitle("Car Tracking Tool")
        icon_path = os.path.join(os.path.dirname(__file__), "UMui/UM.jpg")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.setGeometry(200, 200, 1100, 700)

        self.setup_ui()

        self.log_signal.connect(self.append_log)
        self.video_loaded.connect(self.load_video)
        self.tracking_done.connect(self.on_tracking_done)

    # ui
    def setup_ui(self):
        main_layout = QVBoxLayout()
        top_hbox = QHBoxLayout()
        top_hbox.setContentsMargins(30, 10, 100, 10)#调整与窗口边缘距离
        top_hbox.setSpacing(30)
        left_logo = QLabel()
        right_logo = QLabel()

        left_pix_path = os.path.join(os.path.dirname(__file__), "UMui/UMFST.jpg")
        right_pix_path = os.path.join(os.path.dirname(__file__), "UMui/IOTSC.jpg")

        if os.path.exists(left_pix_path):
            left_pix = QPixmap(left_pix_path)
            left_logo.setPixmap(left_pix.scaled(500, 120, Qt.KeepAspectRatio,
                                                Qt.SmoothTransformation))
        if os.path.exists(right_pix_path):
            right_pix = QPixmap(right_pix_path)
            right_logo.setPixmap(right_pix.scaled(330, 120, Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation))

        top_hbox.addWidget(left_logo, alignment=Qt.AlignLeft)
        top_hbox.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        top_hbox.addWidget(right_logo, alignment=Qt.AlignRight)

        main_layout.addLayout(top_hbox)

        mid_hbox = QHBoxLayout()

        left_vbox = QVBoxLayout()

        # 输入路径
        self.input_label = QLabel("Input video folder:")
        self.input_edit = QLineEdit()
        self.input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input_button = QPushButton("Select")
        self.input_button.clicked.connect(self.select_input_dir)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.input_label)
        hbox1.addWidget(self.input_edit)
        hbox1.addWidget(self.input_button)
        left_vbox.addLayout(hbox1)

        # 输出路径
        self.output_label = QLabel("Output video folder:")
        self.output_edit = QLineEdit()
        self.input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.output_button = QPushButton("Select")
        self.output_button.clicked.connect(self.select_output_dir)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.output_label)
        hbox2.addWidget(self.output_edit)
        hbox2.addWidget(self.output_button)
        left_vbox.addLayout(hbox2)

        # 权重路径
        self.weights_label = QLabel("Model weights(.pt):")
        self.weights_edit = QLineEdit()
        self.input_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.weights_button = QPushButton("Select")
        self.weights_button.clicked.connect(self.select_weights)
        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.weights_label)
        hbox3.addWidget(self.weights_edit)
        hbox3.addWidget(self.weights_button)
        left_vbox.addLayout(hbox3)

        # 开始按钮
        self.run_button = QPushButton("Start tracking")
        self.run_button.setStyleSheet("font-size:16px; font-weight:bold; padding:6px;")
        self.run_button.clicked.connect(self.start_tracking)
        left_vbox.addWidget(self.run_button)

        # 日志框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background:#111; color:#0f0; font-family:Consolas;")
        left_vbox.addWidget(self.log_text, stretch=3)

        # 视频播放组件
        right_vbox = QVBoxLayout()
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(320, 240)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setStyleSheet("background-color:black; border:2px solid #666;")

        # 媒体播放器
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)

        self.audio_output.setVolume(0.5)

        # 调试输出
        self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
        self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.media_player.errorOccurred.connect(self.on_media_error)

        self.play_button = QPushButton("Play / Pause")
        self.play_button.setEnabled(False)
        self.play_button.setStyleSheet("font-size:16px; font-weight:bold; padding:6px;")
        self.play_button.clicked.connect(self.toggle_playback)

        right_vbox.addWidget(self.play_button, alignment=Qt.AlignCenter)
        right_vbox.addWidget(self.video_widget, stretch=1)

        mid_hbox.addLayout(left_vbox, 3)
        mid_hbox.addLayout(right_vbox, 3)
        main_layout.addLayout(mid_hbox)
        self.setLayout(main_layout)

    def select_input_dir(self):
        dirpath = QFileDialog.getExistingDirectory(self, "Select input video folder")
        if dirpath:
            self.input_edit.setText(dirpath)

    def select_output_dir(self):
        dirpath = QFileDialog.getExistingDirectory(self, "Select output video folder")
        if dirpath:
            self.output_edit.setText(dirpath)

    def select_weights(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select model weights", "", "YOLO Weights (*.pt)"
        )
        if filepath:
            self.weights_edit.setText(filepath)

    def start_tracking(self):
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        weights_path = self.weights_edit.text().strip()

        if not os.path.isdir(input_dir):
            QMessageBox.warning(self, "ERROR", "Please select a valid input video folder")
            return
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, "ERROR", "Please select a valid output video folder")
            return
        if not os.path.isfile(weights_path):
            QMessageBox.warning(self, "ERROR", "Please select a valid model weights")
            return

        self.media_player.stop()
        self.video_widget.update()
        self.play_button.setEnabled(False)
        self.play_button.setText("Play")
        self.last_video_path = None
        self.generated_videos = []
        self.media_ended = False

        self.log_text.append("Start processing the videos...\n")
        self.run_button.setEnabled(False)

        thread = threading.Thread(target=self.run_in_thread, daemon=True)
        thread.start()

    def run_in_thread(self):
        import io
        import traceback

        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        try:
            try:
                videos = core_main(
                    input_dir=self.input_edit.text().strip(),
                    output_dir=self.output_edit.text().strip(),
                    weights_path=self.weights_edit.text().strip()
                )
                self.generated_videos = videos or []
                if self.generated_videos:
                    self.last_video_path = random.choice(self.generated_videos)
                    self.log_signal.emit(
                        f"[INFO] Number of output videos {len(self.generated_videos)} 个，"
                        f"Randomly select: {os.path.basename(self.last_video_path)}\n"
                    )
                    self.video_loaded.emit(self.last_video_path)
                else:
                    self.log_signal.emit("[INFO] No video has been output\n")
            except SystemExit:
                return
        except Exception:
            err = traceback.format_exc()
            self.log_signal.emit(f"ERROR:\n{err}")
        finally:
            sys.stdout = old_stdout
            output_text = mystdout.getvalue()
            if output_text:
                self.log_signal.emit(output_text)
            self.tracking_done.emit()

    @Slot(str)
    def append_log(self, text: str):
        self.log_text.append(text)

    @Slot(str)
    def load_video(self, path: str):
        if not path:
            self.log_text.append("[WARN] load_video Empty Path\n")
            return

        abspath = os.path.abspath(path)
        if os.path.exists(abspath):
            url = QUrl.fromLocalFile(abspath)
            self.media_player.setSource(url)
            self.last_video_path = abspath
            self.play_button.setEnabled(True)

            self.media_ended = False

            self.media_player.play()
            self.play_button.setText("Pause")

            self.log_text.append(f"[INFO] Done: {abspath}\n")
        else:
            self.log_text.append(f"[WARN] Invalid video path: {abspath}\n")

    @Slot()
    def on_tracking_done(self):
        self.run_button.setEnabled(True)

    def toggle_playback(self):
        if not self.generated_videos:
            QMessageBox.information(self, "NOTE", "No videos to play. Please run tracking first.")
            return

        if self.media_player.source().isEmpty() or self.media_ended or not self.last_video_path:
            self.play_random_video()
            return

        state = self.media_player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")

    def play_random_video(self):
        if not self.generated_videos:
            QMessageBox.information(self, "NOTE", "No videos from this run to play.")
            return

        choices = list(self.generated_videos)
        if len(choices) > 1 and self.last_video_path in choices:
            choices = [p for p in choices if os.path.abspath(p) != os.path.abspath(self.last_video_path)]

        selected_path = random.choice(choices)
        abspath = os.path.abspath(selected_path)
        if not os.path.exists(abspath):
            self.log_text.append(f"[WARN] Video does not exist.: {abspath}\n")
            return

        url = QUrl.fromLocalFile(abspath)
        self.media_player.setSource(url)
        self.last_video_path = abspath
        self.media_ended = False

        self.media_player.play()
        self.play_button.setText("Pause")

        self.log_text.append(f"[INFO] Play: {os.path.basename(abspath)}\n")

    @Slot(object)
    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            dur = self.media_player.duration()
            if dur > 0:
                pos = max(0, dur - 100)
                self.media_player.pause()
                self.media_player.setPosition(pos)
            self.media_ended = True
            self.play_button.setText("Play")
    @Slot(object)
    def on_playback_state_changed(self, state):
        pass

    @Slot(object)
    def on_media_error(self, error):
        msg = self.media_player.errorString()
        self.log_text.append(f"[MEDIA ERROR] code={error}, msg={msg}\n")
        print("MEDIA ERROR:", error, msg, flush=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TrackerGUI()
    gui.show()
    sys.exit(app.exec())