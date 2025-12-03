from django.contrib import messages
from django.shortcuts import render

# Create your views here.
from django_pandas.io import read_frame
from nltk.corpus import wordnet

from admins.models import storedatamodel
from users.forms import UserRegistrationForm
from users.models import cropyieldUserRegistrationModel, cropyieldanalysismodel
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import numpy as nm
import pandas as pd
import matplotlib.pyplot as mtp


def index(request):
    return render(request,'index.html')


def UserLogin(request):
    return render(request,'users/UserLogin.html',{})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request,'users/UserRegisterForm.html',{'form':form})


def UserRegisterAction(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            # return HttpResponseRedirect('./CustLogin')
            form = UserRegistrationForm()
            return render(request, 'users/UserRegisterForm.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'users/UserRegisterForm.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = cropyieldUserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'users/UserLogin.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'users/UserLogin.html', {})


def UserSendCrop(request):
    return render(request,'users/UserSendCrop.html')


class NaiveBayes:
    def __init__(self, name, crops):
        self.name = name
        self.crops = crops

known_yeilds = [
    NaiveBayes('116.58', set("Rice|andhrapradesh|Kharif".split("|"))),
    NaiveBayes('131.68', set("Rice|westbengal|Kharif".split("|"))),
    NaiveBayes('116.20', set("Rice|uttarpradesh|Kharif".split("|"))),
    NaiveBayes('86.03', set("Rice|punjab|Kharif".split("|"))),
    NaiveBayes('55.97', set("Rice|bihar|Kharif".split("|"))),
    NaiveBayes('50.52', set("Rice|orissa|Kharif".split("|"))),
    NaiveBayes('41.44', set("Rice|chhattisgarh|Kharif".split("|"))),
    NaiveBayes('17.92', set("Rice|assam|Kharif".split("|"))),
    NaiveBayes('76.31', set("Rice|tamilnadu|Kharif".split("|"))),
    NaiveBayes('25.68', set("Rice|haryana|Kharif".split("|"))),
    NaiveBayes('3,322', set("Rice|westgodawari|Kharif".split("|"))),
    NaiveBayes('3,239', set("Rice|guntur|Kharif".split("|"))),
    NaiveBayes('3,142', set("Rice|krishna|Kharif".split("|"))),
    NaiveBayes('2,985', set("Rice|prakasham|Kharif".split("|"))),
    NaiveBayes('2,978', set("Rice|eastgodavari|Kharif".split("|"))),
    NaiveBayes('2,942', set("Rice|kurnool|Kharif".split("|"))),
    NaiveBayes('2,864', set("Rice|nellore|Kharif".split("|"))),
    NaiveBayes('2,630', set("Rice|anantpur|Kharif".split("|"))),
    NaiveBayes('2,610', set("Rice|cuddapah|Kharif".split("|"))),
    NaiveBayes('2,373', set("Rice|chittor|Kharif".split("|"))),
    NaiveBayes('1,957', set("Rice|vizianagaram|Kharif".split("|"))),
    NaiveBayes('1,864', set("Rice|srikakulam|Kharif".split("|"))),
    NaiveBayes('1,430', set("Rice|vishakhapatnam|Kharif".split("|"))),
    NaiveBayes('2,803', set("Rice|karimnagar|Kharif".split("|"))),
    NaiveBayes('2,678', set("Rice|nizamabad|Kharif".split("|"))),
    NaiveBayes('2,578', set("Rice|khammam|Kharif".split("|"))),
    NaiveBayes('3,206', set("Rice|nalgonda|Kharif".split("|"))),
    NaiveBayes('2,398', set("Rice|medak|Kharif".split("|"))),
    NaiveBayes('2,321', set("Rice|rangareddy|Kharif".split("|"))),
    NaiveBayes('2,320', set("Rice|adilabad|Kharif".split("|"))),
    NaiveBayes('3,462', set("Rice|koppal|Kharif".split("|"))),
    NaiveBayes('3,379', set("Rice|davangere|Kharif".split("|"))),
    NaiveBayes('3,247', set("Rice|bellary|Kharif".split("|"))),
    NaiveBayes('2,993', set("Rice|mysore|Kharif".split("|"))),
    NaiveBayes('2,851', set("Rice|raichur|Kharif".split("|"))),
    NaiveBayes('2,749', set("Rice|bangalore|Kharif".split("|"))),
    NaiveBayes('637', set("Rice|bidar|Kharif".split("|"))),
    NaiveBayes('4,574', set("Rice|madurai|Kharif".split("|"))),
    NaiveBayes('4,434', set("Rice|thirunelveli|Kharif".split("|"))),
    NaiveBayes('3,769', set("Rice|vellore|Kharif".split("|"))),
    NaiveBayes('4,322', set("Rice|palamaner|Kharif".split("|"))),
    NaiveBayes('3,712', set("Rice|palamaner|Rabi".split("|"))),
    NaiveBayes('2,213', set("Wheat|palamaner|Kharif".split("|"))),
    NaiveBayes('234', set("Wheat|palamaner|Rabi".split("|"))),
    NaiveBayes('4863', set("Rice|chittoor|Kharif".split("|"))),
    NaiveBayes('2653', set("Rice|chittoor|Rabi".split("|"))),












]



def UserSendCropanalysis(request):
    if request.method == "POST":
        crops = request.POST.get('crop')
        print(crops)
        loginid = request.POST.get('loginid')
        print(loginid)
        try:

            check = cropyieldUserRegistrationModel.objects.get(loginid=loginid)
            loginid = check.loginid
            print("name", loginid)
            email = check.email
            storcrops = crops
            print(check.email, storcrops)
            crops = crops.lower()
            #print("crops:",crops)
            crops = crops.split(",")
            possible = []
            for crop in crops:
                #print("crop",crop)
                for yeilds in known_yeilds:
                    #print("yeilds",yeilds)
                    if crop in yeilds.crops:
                        possible.append(yeilds.name)
            if possible:
                print("possible",possible)
                for x in possible:
                    print('yeild is = ', x)
                    #recDescription = recDesc[x]

                    ing = wordnet.synsets(x)
                    description = ''
                    if len(ing) != 0:
                        description = ing[0].definition()
                        print(description)
                    else:
                        description = 'No Data found'
                        cropyieldanalysismodel.objects.create(loginid=loginid, email=email, cropdetails=storcrops,
                                                              yields=x)
                messages.success(request, 'Your Request Sent to admin')
            else:
                messages.success(request, "Sorry,Based on details we can't provide proper deatil")
            return render(request, 'users/UserSendCrop.html')
        except Exception  as e:
            print(str(e))

        messages.success(request, 'There is a problam in your details')
        return render(request, 'users/UserSendCrop.html')


def yeilddetails(request):
    email = request.session['email']
    sts = 'sent'
    dict = cropyieldanalysismodel.objects.filter(email=email,status=sts).order_by('-id')
    return render(request,'users/yeildsdetails.html',{'data':dict})


def ML(request):
    qs = storedatamodel.objects.all()
    data = read_frame(qs)
    data = data.fillna(data.mean())
    # data[0:label]
    data.info()
    print(data.head())
    print(data.describe())
    #print(data.shape)
    # print("data-label:",data.label)
    dataset = data.iloc[:,[3,4]].values
    print("x", dataset)
    dataset1 = data.iloc[:,-1].values
    print("y", dataset1)
    print("shape", dataset.shape)
    X = dataset
    y = dataset1
    print(dataset.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
    st_X = StandardScaler()
    X_train = st_X.fit_transform(X_train)
    X_test = st_X.transform(X_test)
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("y_pred", y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("cm", cm)
    X_set, y_set = X_train, y_train
    X1, X2 = nm.meshgrid(nm.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         nm.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    mtp.contourf(X1, X2, classifier.predict(nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    mtp.xlim(X1.min(), X1.max())
    mtp.ylim(X2.min(), X2.max())
    for i, j in enumerate(nm.unique(y_set)):
        mtp.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    mtp.title('K-NN Algorithm (Training set)')
    mtp.xlabel('Year')
    mtp.ylabel('Estimated yeild')
    mtp.legend()
    mtp.show()
    return render(request, 'users/UserHomePage.html')