from PyQt5.QtWidgets import QWidget, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon

from os import listdir
from os import path as os_path

from torch.nn import Linear, ReLU, Module, CrossEntropyLoss, Softmax
from torch.nn.init import xavier_uniform_, zeros_, calculate_gain, kaiming_uniform_
from torch import device, load, no_grad, from_numpy, Tensor, mm, save
from torch import max as torch_max
from torch.optim import Adam

from numpy import round as np_round
from numpy import sum as np_sum
from numpy import absolute as np_absolute
from numpy import mean as np_mean
from numpy import load as np_load
from numpy import array as np_array
from numpy import savez as np_savez
from numpy import shape as np_shape
from numpy import squeeze as np_squeeze
from numpy.random import shuffle
from numpy import arange as np_arange
from numpy import corrcoef as np_corrcoef
from cv2 import imread, cvtColor, COLOR_BGR2HSV, split, calcHist, resize
from time import time

device = device('cpu')
input_size = 7
hidden_size = 150
num_classes = 4
num_loop_epoch = 50
num_epochs = 5000
learning_rate = 0.0003
WIDTH = 4500
HEIGHT = 3000

class NeuralNet(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = Linear(input_size, hidden_size) 
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)
        self.softmax = Softmax(-1)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
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
        self.TrainingTabUI()

    def TrainingTabUI(self):
        self.trainingWidget = QWidget(self)
        self.trainingWidget.setGeometry(self.left, self.top, self.width, self.height)
        
        # GROUP STAGE 60
        # Create a button in the window
        self.folderImageButton60 = QPushButton('Thư mục ảnh huấn luyện giai đoạn 60 phút', self.trainingWidget)
        self.folderImageButton60.setGeometry(self.left + 10, self.height//7, self.width//3, self.height//10)
        self.folderImageButton60.setFont(self.font14)
        self.folderImageButton60.setStyleSheet("background-color: yellow; font-weight: bold")
        self.folderImageButton60.setIcon(QIcon("folder.png"))
        # Button training 60
        self.training60 = QPushButton("Train", self.trainingWidget)
        self.training60.setGeometry(self.width//3 + 30, self.height//7, self.width//5, self.height//10)
        self.training60.setFont(self.font18)
        self.training60.setStyleSheet("color: black; font-weight: bold")
        self.training60.clicked.connect(lambda: self.on_click("60"))

        # GROUP STAGE 90
        # Create a button in the window
        self.folderImageButton90 = QPushButton('Thư mục ảnh huấn luyện giai đoạn 90 phút', self.trainingWidget)
        self.folderImageButton90.setGeometry(self.left + 10, 3*self.height//7, self.width//3, self.height//10)
        self.folderImageButton90.setFont(self.font14)
        self.folderImageButton90.setStyleSheet("background-color: yellow; font-weight: bold")
        self.folderImageButton90.setIcon(QIcon("folder.png"))
        # Button training 90
        self.training90 = QPushButton("Train", self.trainingWidget)
        self.training90.setGeometry(self.width//3 + 30, 3*self.height//7, self.width//5, self.height//10)
        self.training90.setFont(self.font18)
        self.training90.setStyleSheet("color: black; font-weight: bold")
        self.training90.clicked.connect(lambda: self.on_click("90"))

        # GROUP STAGE 120
        # Create a button in the window
        self.folderImageButton120 = QPushButton('Thư mục ảnh huấn luyện giai đoạn 120 phút', self.trainingWidget)
        self.folderImageButton120.setGeometry(self.left + 10, 5*self.height//7, self.width//3, self.height//10)
        self.folderImageButton120.setFont(self.font14)
        self.folderImageButton120.setStyleSheet("background-color: yellow; font-weight: bold")
        self.folderImageButton120.setIcon(QIcon("folder.png"))
        # Button training 120
        self.training120 = QPushButton("Train", self.trainingWidget)
        self.training120.setGeometry(self.width//3 + 30, 5*self.height//7, self.width//5, self.height//10)
        self.training120.setFont(self.font18)
        self.training120.setStyleSheet("color: black; font-weight: bold")
        self.training120.clicked.connect(lambda: self.on_click("120"))

    def on_click(self, groupStage):
        path_dataset = os_path.join("./CUSTOMIZE_4_USER/TRAINING/Data/", groupStage)
        img1, img2 = self.preprocess_datasets(path_dataset, groupStage)
        if img1==-1 and img2 ==-1:
            QMessageBox.about(self, "Warning", "Thư mục ảnh mẫu không đủ")
            return
        ret = self.extracting_feature(path_dataset, img1, img2, groupStage) 
        if ret == -1:
            QMessageBox.about(self, "Warning", "Thư mục ảnh mẫu không đủ")
            return
        self.training_data(groupStage)


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

        img0 = path_img[save[0]]
        img1 = path_img[save[1]]

        imgSample1 = os_path.join(PATH_DATA, img0)
        imgSample2 = os_path.join(PATH_DATA, img1)
        return  imgSample1, imgSample2

    def extracting_feature(self, path_dataset, imgSample1, imgSample2, groupStage):
        PATH_FEATURE_MODEL = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+ ".npz")
        feature = []
        labels = []
        print("(INFO) EXTRACTING FEATURE ...")
        def process_feature(list_path, labelFeature):
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

        np_savez(PATH_FEATURE_MODEL, data_max = hist_max, data_min = hist_min, ColourFeature = feature, Labels = labels)

    def training_data(self, groupStage):
        print("(INFO) START TRAINING STAGE {} ! ".format(groupStage))
        # Path to extracted feature
        feature_path = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+".npz")
        model_path = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+".pth")
        data = np_load(feature_path)
        feature = data['ColourFeature']
        feature.astype(int)
        label = data['Labels']
        label = np_squeeze(label)

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
        
        X_train = from_numpy(X_train).to(device).float()
        y_train = from_numpy(y_train).to(device).long()

        X_test = from_numpy(X_test).to(device).float()
        y_test = from_numpy(y_test).to(device).long()
        
        model = NeuralNet(input_size, hidden_size, num_classes).to(device)

        # Loss and optimizer
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        max_acc = 30
        for k in range(num_loop_epoch):
            model.apply(self.weight_init)
            for epoch in range(num_epochs):
                # Forward pass
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (epoch+1)%1000 == 0:
                    print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, loss.item()))

            with no_grad():
                out_test = model(X_test)
                _, predicted = torch_max(out_test.data, 1)
                corrected = (predicted==y_test).sum().item()
                total = len(y_test)
                accuracy = 100*(corrected/total)
                print("accuracy: {} %".format(accuracy))
            if accuracy > max_acc:
                max_acc = accuracy
                save(model.state_dict(), model_path)
                print("Save this model " + str(groupStage) + ": " + str(accuracy) )
        if max_acc==30:
            print("Bad training!!!")

    # Thif function reset parameter of model
    def weight_init(self, m):
        if isinstance(m, Linear):
            #xavier_uniform_(m.weight, gain=calculate_gain('relu'))
            kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            m.bias.data.fill_(0.01)
            # zeros_(m.bias)


def predict_image(path_of_image, groupStage):
    path_of_model = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+".pth")
    path_of_feature = os_path.join("./CUSTOMIZE_4_USER/MODEL_TRAINING", groupStage, groupStage+".npz")

    start_time = time()
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
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
    image_feature = from_numpy(image_feature).to(device).float().view(1, input_size)

    with no_grad():
        out_predict = model(image_feature)
        _, predicted_result = torch_max(out_predict.data, 1)
        original = Tensor([[1, 33, 66, 99]])
    
    # Round xx.xx %
    percentage_result = np_round(mm(out_predict.view(1, num_classes), original.view(num_classes, 1)).item(), 2)
    
    # Processed time 
    processedTime = np_round(time()-start_time, 2)
    #print("Time  ",processedTime)
  
    return percentage_result, processedTime
