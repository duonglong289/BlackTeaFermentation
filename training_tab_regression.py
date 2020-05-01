from PyQt5.QtWidgets import QWidget, QPushButton, QMessageBox, QFileDialog, QLineEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QTimer

from os import listdir, remove
from os import path as os_path
from os import environ

from torch.nn import Linear, ReLU, Module, MSELoss
from torch.nn.init import xavier_uniform_, zeros_, calculate_gain, kaiming_uniform_
from torch import device, load, no_grad, from_numpy, Tensor, mm, save
from torch import max as torch_max
from torch.optim import Adam
from torch import cuda

from numpy import round as np_round
from numpy import sum as np_sum
from numpy import absolute as np_absolute
from numpy import mean as np_mean
from numpy import load as np_load
from numpy import array as np_array
from numpy import savez as np_savez
from numpy import shape as np_shape
from numpy import squeeze as np_squeeze
from numpy.random import shuffle, seed
from numpy import arange as np_arange
from numpy import corrcoef as np_corrcoef
from cv2 import imread, cvtColor, COLOR_BGR2HSV, split, calcHist, resize
from time import time
from shutil import copy


# Select cuda or cpu device
def select_device(device_select='', apex=False, batch_size=None):
    cpu_request = device_select.lower() == 'cpu'
    if device_select and not cpu_request:  # if device requested other than 'cpu'
        environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device_select  # check availablity

    cuda_device = False if cpu_request else cuda.is_available()
    if cuda_device:
        c = 1024 ** 2  # bytes to MB
        ng = cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return device('cuda:0' if cuda_device else 'cpu')


DEVICE = select_device()
input_size = 7
# hidden_size = 150
num_classes = 1
num_loop_epoch = 10
num_epochs = 2000
learning_rate = 1e-1
WIDTH = 4500
HEIGHT = 3000
seed(2)

class NeuralNet(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = Linear(input_size, hidden_size) 
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)
        # self.softmax = Softmax(-1)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.softmax(out)
        return out

class TrainingTab(QWidget):
    def __init__(self, parent):
        super(TrainingTab, self).__init__(parent)
        self.left = 0
        self.top = 0
        self.height = parent.height
        self.width = parent.width
        self.font18 = parent.font18
        self.font14 = parent.font14
        self.status = False
        self.TrainingTabUI()

    def TrainingTabUI(self):
        self.trainingWidget = QWidget(self)
        self.trainingWidget.setGeometry(self.left, self.top, self.width, self.height)
        
        # GROUP STAGE 60
        # Create a button in the window
        self.trainingStage60 = QPushButton('Huấn luyện giai đoạn 60 phút', self.trainingWidget)
        self.trainingStage60.setGeometry(self.left + 10, self.height//7, self.width//3, self.height//10)
        self.trainingStage60.setFont(self.font14)
        self.trainingStage60.setStyleSheet("background-color: yellow; font-weight: bold")
        self.trainingStage60.setCheckable(True)
        self.trainingStage60.toggled[bool].connect(lambda:self.choosingTrainingStage("60"))

        # GROUP STAGE 90
        # Create a button in the window
        self.trainingStage90 = QPushButton('Huấn luyện giai đoạn 90 phút', self.trainingWidget)
        self.trainingStage90.setGeometry(self.left + 10, 3*self.height//7, self.width//3, self.height//10)
        self.trainingStage90.setFont(self.font14)
        self.trainingStage90.setStyleSheet("background-color: yellow; font-weight: bold")
        self.trainingStage90.setCheckable(True)
        self.trainingStage90.toggled[bool].connect(lambda:self.choosingTrainingStage("90"))
        
        # GROUP STAGE 120
        # Create a button in the window
        self.trainingStage120 = QPushButton('Huấn luyện giai đoạn 120 phút', self.trainingWidget)
        self.trainingStage120.setGeometry(self.left + 10, 5*self.height//7, self.width//3, self.height//10)
        self.trainingStage120.setFont(self.font14)
        self.trainingStage120.setStyleSheet("background-color: yellow; font-weight: bold")
        self.trainingStage120.setCheckable(True)
        self.trainingStage120.toggled[bool].connect(lambda:self.choosingTrainingStage("120"))
    

        # Button training 
        self.trainingButton = QPushButton("Train", self.trainingWidget)
        self.trainingButton.setGeometry(2*self.width//3 , 3*self.height//7, self.width//6, self.height//10)
        self.trainingButton.setFont(self.font18)
        self.trainingButton.setStyleSheet("color: black; font-weight: bold")
        self.trainingButton.clicked.connect(self.excecuteTraining)

        # Timer check training status
        self.timerTraining = QTimer()
        self.timerTraining.timeout.connect(self.runTraining)
        self.timerTraining.setInterval(1000)
        self.timerTraining.start()

        # Status training
        self.sttTraining = QLineEdit(self.trainingWidget)
        self.sttTraining.setGeometry(2*self.width//3 , 4*self.height//7, self.width//6, self.height//10)
        self.sttTraining.setFont(self.font14)
        self.sttTraining.setAlignment(Qt.AlignCenter)
        self.sttTraining.setReadOnly(True)
        self.sttTraining.setStyleSheet("border: 2px solid gray; border-radius: 10px; padding:0 8px;")
        
        
        # folder 0-50%:
        self.buttonFolder1 = QPushButton("Chọn thư mục ảnh giai đoạn 0-25%", self.trainingWidget)
        self.buttonFolder1.setGeometry(self.width//3+30, 1*self.height//9, self.width//5+40, self.height//12)
        self.buttonFolder1.setFont(self.font14)
        self.buttonFolder1.setIcon(QIcon("folder.png"))
        self.buttonFolder1.clicked.connect(lambda: self.browseImage("1"))
        # folder 50-70%
        self.buttonFolder2 = QPushButton("Chọn thư mục ảnh giai đoạn 25-50%", self.trainingWidget)
        self.buttonFolder2.setGeometry(self.width//3+30, 3*self.height//9, self.width//5+40, self.height//12)
        self.buttonFolder2.setFont(self.font14)
        self.buttonFolder2.setIcon(QIcon("folder.png"))
        self.buttonFolder2.clicked.connect(lambda: self.browseImage("2"))
        # folder 70-90 %
        self.buttonFolder3 = QPushButton("Chọn thư mục ảnh giai đoạn 50-75%", self.trainingWidget)
        self.buttonFolder3.setGeometry(self.width//3+30, 5*self.height//9, self.width//5+40, self.height//12)
        self.buttonFolder3.setFont(self.font14)
        self.buttonFolder3.setIcon(QIcon("folder.png"))
        self.buttonFolder3.clicked.connect(lambda: self.browseImage("3"))
        # folder >90%
        self.buttonFolder4 = QPushButton("Chọn thư mục ảnh giai đoạn 75-100%", self.trainingWidget)
        self.buttonFolder4.setGeometry(self.width//3+30, 7*self.height//9, self.width//5+40, self.height//12)
        self.buttonFolder4.setFont(self.font14)
        self.buttonFolder4.setIcon(QIcon("folder.png"))
        self.buttonFolder4.clicked.connect(lambda: self.browseImage("4"))

    # Timer execute training stage
    def runTraining(self):
        if self.status == True:
            self.trainingFunction()
        else:
            return

    # Clicked training button
    def excecuteTraining(self):
        self.timerTraining.start()
        self.status = True
        self.sttTraining.setText("Đang huấn luyện...")

    # Browse image folder
    def browseImage(self, grade):
        if self.trainingStage60.isChecked()==False and self.trainingStage90.isChecked()==False and self.trainingStage120.isChecked()==False:
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        if grade=="1":
            self.imageFolderGrade1 = QFileDialog.getExistingDirectory(self, "Nhập đường dẫn thư mục ảnh giai đoạn 1")
        if grade=="2":
            self.imageFolderGrade2 = QFileDialog.getExistingDirectory(self, "Nhập đường dẫn thư mục ảnh giai đoạn 2")
        if grade=="3":
            self.imageFolderGrade3 = QFileDialog.getExistingDirectory(self, "Nhập đường dẫn thư mục ảnh giai đoạn 3")
        if grade=="4":
            self.imageFolderGrade4 = QFileDialog.getExistingDirectory(self, "Nhập đường dẫn thư mục ảnh giai đoạn 4")
        
    # Choose training stage button and disable other stage button
    def choosingTrainingStage(self, groupStage):
        if groupStage=="60":
            if self.trainingStage60.isChecked()==True:
                self.trainingStage90.setCheckable(False)
                self.trainingStage90.setStyleSheet("background-color: gray")
                self.trainingStage120.setCheckable(False)
                self.trainingStage120.setStyleSheet("background-color: gray")
                self.imageFolderGrade1 = ""
                self.imageFolderGrade2 = ""
                self.imageFolderGrade3 = ""
                self.imageFolderGrade4 = ""

            if self.trainingStage60.isChecked()==False:
                self.trainingStage90.setCheckable(True)
                self.trainingStage90.setStyleSheet("background-color: yellow; font-weight: bold")
                self.trainingStage120.setCheckable(True)
                self.trainingStage120.setStyleSheet("background-color: yellow; font-weight: bold")

        if groupStage == "90":   
            if self.trainingStage90.isChecked()==True:
                self.trainingStage120.setCheckable(False)
                self.trainingStage120.setStyleSheet("background-color: gray")
                self.trainingStage60.setCheckable(False)
                self.trainingStage60.setStyleSheet("background-color: gray")
                self.imageFolderGrade1 = ""
                self.imageFolderGrade2 = ""
                self.imageFolderGrade3 = ""
                self.imageFolderGrade4 = ""

            if self.trainingStage90.isChecked()==False:
                self.trainingStage60.setCheckable(True)
                self.trainingStage60.setStyleSheet("background-color: yellow; font-weight: bold")
                self.trainingStage120.setCheckable(True)
                self.trainingStage120.setStyleSheet("background-color: yellow; font-weight: bold")

        if groupStage == "120":
            if self.trainingStage120.isChecked()==True:
                self.trainingStage90.setCheckable(False)
                self.trainingStage90.setStyleSheet("background-color: gray")
                self.trainingStage60.setCheckable(False)
                self.trainingStage60.setStyleSheet("background-color: gray")
                self.imageFolderGrade1 = ""
                self.imageFolderGrade2 = ""
                self.imageFolderGrade3 = ""
                self.imageFolderGrade4 = ""

            if self.trainingStage120.isChecked()==False:
                self.trainingStage90.setCheckable(True)
                self.trainingStage90.setStyleSheet("background-color: yellow; font-weight: bold")
                self.trainingStage60.setCheckable(True)
                self.trainingStage60.setStyleSheet("background-color: yellow; font-weight: bold")


    def trainingFunction(self):
        # Check model stage and return if no of models were clicked
        if self.trainingStage60.isChecked()==False and self.trainingStage90.isChecked()==False and self.trainingStage120.isChecked()==False:
            self.sttTraining.setText("Chưa chọn mô hình.")
            self.status = False
            return
        
        if self.trainingStage60.isChecked()==True:
            groupStage = "60"
            ret = self.checkFolderTraining(groupStage)
        if self.trainingStage90.isChecked()==True:
            groupStage = "90"
            ret = self.checkFolderTraining(groupStage)
        if self.trainingStage120.isChecked()==True:
            groupStage = "120"
            ret = self.checkFolderTraining(groupStage)
        
        # return if image path==""
        if ret==-1:
            self.sttTraining.setText("Chưa chọn ảnh.")
            self.status = False
            return
        
        # return if at least 1 folder is empty
        path_dataset = os_path.join("./CUSTOMIZE_4_USER/TRAINING/Data/", groupStage)
        img1, img2 = self.preprocess_datasets(path_dataset, groupStage)
        if img1==-1 and img2 ==-1:
            QMessageBox.about(self, "Warning", "Thư mục ảnh mẫu không đủ")
            self.status = False
            self.sttTraining.setText("Chưa đủ ảnh.")
            return

        ret = self.extracting_feature(path_dataset, img1, img2, groupStage) 
        if ret == -1:
            QMessageBox.about(self, "Warning", "Thư mục ảnh mẫu không đủ")
            self.status = False
            self.sttTraining.setText("Chưa đủ ảnh.")
            return

        # self.sttTraining.setText("Đang huấn luyện...")
        self.training_data(groupStage)

    # Copy all images to default source folder
    def checkFolderTraining(self, groupStage):
        if self.imageFolderGrade1 =="" or self.imageFolderGrade2 =="" or self.imageFolderGrade3 =="" or self.imageFolderGrade4 =="":
            QMessageBox.about(self,"Warning","Chưa chọn đủ dữ liệu huấn luyện.")
            return -1
        # Copy all training images to default folder
        print("(INFO) COPYING IMAGES ...")
        def copyImageTraining(src, grade):
            path_dataset = os_path.join("./CUSTOMIZE_4_USER/TRAINING/Data/", groupStage, grade)
            dst_files = listdir(path_dataset)
            # remove old images
            for filesrc in dst_files:
                remove(os_path.join(path_dataset, filesrc))
            # List all images ends with "jpg", "png", "jpeg"
            src_files = listdir(src)
            for file_name in src_files:
                if not(file_name.endswith(".jpg") or file_name.endswith(".JPG") or file_name.endswith(".png") or file_name.endswith(".PNG") or file_name.endswith(".jpeg") or file_name.endswith(".JPEG")):
                    continue
                full_file_name = os_path.join(src, file_name)
                if os_path.isfile(full_file_name):
                    copy(full_file_name, path_dataset)

        copyImageTraining(self.imageFolderGrade1, "1")
        copyImageTraining(self.imageFolderGrade2, "2")
        copyImageTraining(self.imageFolderGrade3, "3")
        copyImageTraining(self.imageFolderGrade4, "4")

    # Get 2 featured images
    def preprocess_datasets(self, path_dataset, groupStage):
        PATH_DATA = os_path.join(path_dataset, "4")
        print("(INFO) EVALUATING DATASET ...")
        path_img = sorted(listdir(PATH_DATA))
        if path_img==[]:
            return -1,-1
        num_img = len(path_img)

        # Histogram of all images in folder
        hChannel = []
        sChannel = []
        vChannel = []

        for image_path in path_img:
            img = imread(os_path.join(PATH_DATA, image_path))
            img = resize(img, (6000,4000))
            img = img[500:-500, 750:-750, :]
            # HSV channel
            img = cvtColor(img, COLOR_BGR2HSV)
            # HSV histogram
            h = calcHist([img], [0], None, [256],[0,256]).reshape(256,)
            s = calcHist([img], [1], None, [256],[0,256]).reshape(256,)
            v = calcHist([img], [2], None, [256],[0,256]).reshape(256,)
            
            hChannel.append(h)
            sChannel.append(s)
            vChannel.append(v)

        # Compute dissimilarity 
        maxI = 0
        for i in range(num_img):
            one = []
            for j in range(num_img):
                c1 = np_sum(np_absolute(hChannel[j]-hChannel[i])) / (HEIGHT * WIDTH)
                c2 = np_sum(np_absolute(sChannel[j]-sChannel[i])) / (HEIGHT * WIDTH)
                c = (c1+c2)/2
                if c > maxI:
                    maxI = c
                    save = [i,j]
        # Get 2 featured images
        img0 = path_img[save[0]]
        img1 = path_img[save[1]]
        # Get paths of 2 images
        imgSample1 = os_path.join(PATH_DATA, img0)
        imgSample2 = os_path.join(PATH_DATA, img1)
        
        return  imgSample1, imgSample2

    def extracting_feature(self, path_dataset, imgSample1, imgSample2, groupStage):
        PATH_FEATURE_MODEL = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+ ".npz")
        feature = []
        labels = []
        print("(INFO) EXTRACT FEATURE ...")
        def process_feature(list_path, labelFeature):
            print("Extracting...")
            list_dir = sorted(listdir(list_path))
            if list_dir == []:
                return -1
            for image_path in list_dir:
                name_image = os_path.join(list_path, image_path)
                if name_image == imgSample1 or name_image == imgSample2:
                    continue
                img = imread(name_image)
                img = resize(img, (6000,4000))
                img = img[500:-500, 750:-750, :]
                img = cvtColor(img, COLOR_BGR2HSV)
                hchan, schan, vchan = split(img)
                h_hist = calcHist([img], [0], None, [256], [0,256]).reshape(256,)
                s_hist = calcHist([img], [1], None, [256], [0,256]).reshape(256,)
                v_hist = calcHist([img], [2], None, [256], [0,256]).reshape(256,)
                
                # 7 feature consist of :
                # + Compute mean value pixel of H channel
                # + Dissilarity with H channel of "max" image
                # + Dissilarity with H channel of "min" image
                # + Compute mean value pixel of S channel
                # + Dissilarity with S channel of "max" image
                # + Dissilarity with S channel of "min" image
                # + Correlation between histogram of H and S channel
                hMean = np_mean(hchan)/255
                DPV_h_max = np_sum(np_absolute(h_hist - h_max))/(HEIGHT*WIDTH)
                DPV_h_min = np_sum(np_absolute(h_hist - h_min))/(HEIGHT*WIDTH)
                
                sMean = np_mean(schan)/255
                DPV_s_max = np_sum(np_absolute(s_hist - s_max))/(HEIGHT*WIDTH)
                DPV_s_min = np_sum(np_absolute(s_hist - s_min))/(HEIGHT*WIDTH)
                
                vMean = np_mean(vchan)/255
                DPV_v_max = np_sum(np_absolute(v_hist - v_max))/(HEIGHT*WIDTH)
                DPV_v_min = np_sum(np_absolute(v_hist - v_min))/(HEIGHT*WIDTH)
                
                correlation = np_corrcoef(h_hist, s_hist)[0][1]
                # variable = [hMean, DPV_h_max, DPV_h_min, sMean, DPV_s_max, DPV_s_min, vMean, DPV_v_max, DPV_v_min]
                variable = [hMean, DPV_h_max, DPV_h_min, sMean, DPV_s_max, DPV_s_min, correlation]
                feature.append(variable)
                labels.append([labelFeature])

        img_max = imread(imgSample1)
        img_max = resize(img_max, (6000,4000))
        img_max = img_max[500:-500, 750:-750, :]
        img_max = cvtColor(img_max, COLOR_BGR2HSV)
        h_max = calcHist([img_max], [0], None, [256],[0,256]).reshape(256,)
        s_max = calcHist([img_max], [1], None, [256],[0,256]).reshape(256,)
        v_max = calcHist([img_max], [2], None, [256],[0,256]).reshape(256,)

        img_min = imread(imgSample2)
        img_min = resize(img_min, (6000,4000))
        img_min = img_min[500:-500, 750:-750, :]
        img_min = cvtColor(img_min, COLOR_BGR2HSV)
        h_min = calcHist([img_min], [0], None, [256],[0,256]).reshape(256,)
        s_min = calcHist([img_min], [1], None, [256],[0,256]).reshape(256,)
        v_min = calcHist([img_min], [2], None, [256],[0,256]).reshape(256,)

        hist_max = [h_max, s_max, v_max]
        hist_min = [h_min, s_min, v_min]  

        # 0%
        list_path_1 = os_path.join(path_dataset, "1")
        process_feature(list_path_1, 0)
        # 33%
        list_path_2 = os_path.join(path_dataset, "2")
        process_feature(list_path_2, 1)
        # 66%
        list_path_3 = os_path.join(path_dataset, "3")
        process_feature(list_path_3, 2)
        # 99%
        list_path_4 = os_path.join(path_dataset, "4")
        process_feature(list_path_4, 3)

        feature = np_array(feature)
        labels = np_array(labels)
        hist_max = np_array(hist_max)
        hist_min = np_array(hist_min)
        
        # Save features to "./CUSTOMIZE_4_USER/MODEL_TRAINING/"stage"/"stage.npz"
        np_savez(PATH_FEATURE_MODEL, data_max = hist_max, data_min = hist_min, ColourFeature = feature, Labels = labels)

    def training_data(self, groupStage):
        print("(INFO) START TRAINING STAGE {} ! ".format(groupStage))
        # Path to extracted feature
        feature_path = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+".npz")
        model_path = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+".pth")
        hidden_path = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+"_hidden.txt")
        # Load extracted feature
        data = np_load(feature_path)
        feature = data['ColourFeature']
        feature.astype(int)
        label = data['Labels']/3

        # train - test split data
        num_train = int(0.8 * len(feature))
        num_test = len(feature) - num_train
        arr = np_arange(len(feature))
        shuffle(arr)
        train_num = arr[:num_train]
        test_num = arr[num_train:]

        y_train = label[train_num]
        y_test = label[test_num]

        X_train = feature[train_num]
        X_test = feature[test_num]
        
        X_train = from_numpy(X_train).to(DEVICE).float()
        y_train = from_numpy(y_train).to(DEVICE).float()

        X_test = from_numpy(X_test).to(DEVICE).float()
        y_test = from_numpy(y_test).to(DEVICE).float()
        
        # Number of neuron in hidden layer
        hidden_list = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230]
        min_val = 1

        for hidden_size in hidden_list:
            print("(INFO) TRAINING WITH HIDDEN SIZE: ", str(hidden_size))
            model = NeuralNet(input_size, hidden_size, num_classes).to(DEVICE)
            # Loss and optimizer
            criterion = MSELoss()
            optimizer = Adam(model.parameters(), lr=learning_rate)
            
            for k in range(num_loop_epoch):
                model.apply(self.weight_init)
                for epoch in range(num_epochs):
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

                    with no_grad():
                        out_test = model(X_test)
                        loss_test = criterion(out_test, y_test)
                    if (epoch+1)%1000 == 0:
                        print ('Epoch [{}/{}], Training-loss: {:.4f}, Val_loss: {:.4f}' .format(epoch+1, num_epochs, loss.item(), loss_test.item()))
                
                # Save the best model has smallest val-loss
                # Metric: MSE
                if loss_test.item()< min_val:
                    min_val = loss_test.item()
                    save(model.state_dict(), model_path)
                    with open(hidden_path,"w") as f:
                        f.write(str(hidden_size))

                    print("Save this model " + str(groupStage) + ": " + str(min_val) +" with hidden size:" +str(hidden_size))
        
        # Set status of training process
        self.sttTraining.setText("Hoàn thành!")
        self.status = False
        
    # This function reset parameters of model: He initialization
    def weight_init(self, m):
        if isinstance(m, Linear):
            kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            m.bias.data.fill_(0.01)


# Predict value of fermented tea image
def predict_image(path_of_image, groupStage):
    path_of_model = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+".pth")
    path_of_feature = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+".npz")
    hidden_path = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+"_hidden.txt")
    # if hidden number file is not existing, return 0
    try:
        with open(hidden_path,"r") as f:
            hidden_size = int(f.readline())
    except FileNotFoundError:
        print("ERROR: No hidden number file in folder")
        return 0.0,0
    
    # Calculate processing time 
    start_time = time()
    model = NeuralNet(input_size, hidden_size, num_classes).to(DEVICE)
    model.load_state_dict(load(path_of_model))

    data = np_load(path_of_feature)
    [h_max, s_max, v_max] = data['data_max']
    [h_min, s_min, v_min] = data['data_min']
    
    img = imread(path_of_image)
    img = resize(img, (6000,4000))
    img = img[500:-500, 750:-750, :]
    img = cvtColor(img, COLOR_BGR2HSV)
    hchan, schan, vchan = split(img)
    h_hist = calcHist([img], [0], None, [256], [0,256]).reshape(256,)
    s_hist = calcHist([img], [1], None, [256], [0,256]).reshape(256,)
    v_hist = calcHist([img], [2], None, [256], [0,256]).reshape(256,)
    
    # 7 features consist of :
    # + Compute mean value pixel of H channel
    # + Dissilarity with H channel of "max" image
    # + Dissilarity with H channel of "min" image
    # + Compute mean value pixel of S channel
    # + Dissilarity with S channel of "max" image
    # + Dissilarity with S channel of "min" image
    # + Correlation between histogram of H and S channel
    hMean = np_mean(hchan)/255
    DPV_h_max = np_sum(np_absolute(h_hist - h_max))/(HEIGHT*WIDTH)
    DPV_h_min = np_sum(np_absolute(h_hist - h_min))/(HEIGHT*WIDTH)
    
    sMean = np_mean(schan)/255
    DPV_s_max = np_sum(np_absolute(s_hist - s_max))/(HEIGHT*WIDTH)
    DPV_s_min = np_sum(np_absolute(s_hist - s_min))/(HEIGHT*WIDTH)
    
    vMean = np_mean(vchan)/255
    DPV_v_max = np_sum(np_absolute(v_hist - v_max))/(HEIGHT*WIDTH)
    DPV_v_min = np_sum(np_absolute(v_hist - v_min))/(HEIGHT*WIDTH)

    correlation = np_corrcoef(h_hist, s_hist)[0][1]
    
    #image_feature = np_array((hMean, DPV_h_max, DPV_h_min, sMean, DPV_s_max, DPV_s_min, vMean, DPV_v_max, DPV_v_min))
    image_feature = np_array((hMean, DPV_h_max, DPV_h_min, sMean, DPV_s_max, DPV_s_min, correlation))
    image_feature = from_numpy(image_feature).to(DEVICE).float().view(1, input_size)

    with no_grad():
        out_predict = model(image_feature)
        
    # Round xx.xx %
    percentage_result = np_round(out_predict.item()*99, 2)
    if percentage_result >99.99:
        percentage_result = 99.99

    if percentage_result <1.0:
        percentage_result = 1.0    
    # Processed time 
    processedTime = np_round(time()-start_time, 2)

    return percentage_result, processedTime
