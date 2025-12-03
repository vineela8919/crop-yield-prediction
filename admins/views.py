from django.contrib import messages
from django.shortcuts import render
from io import TextIOWrapper
import csv
from collections import defaultdict
from django.shortcuts import render, HttpResponse
# Create your views here.
from django_pandas.io import read_frame

from admins.models import  storedatamodel
from users.models import cropyieldUserRegistrationModel, cropyieldanalysismodel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import numpy as nm
import pandas as pd
import matplotlib.pyplot as mtp
from sklearn.metrics import classification_report


def AdminLogin(request):
    return render(request,'admins/AdminLogin.html',{})

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'admins/AdminLogin.html', {})


def AdminViewUsers(request):
    data = cropyieldUserRegistrationModel.objects.all()
    return render(request,'admins/AdminViewUsers.html',{'data':data})

def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        cropyieldUserRegistrationModel.objects.filter(id=id).update(status=status)
        data = cropyieldUserRegistrationModel.objects.all()
        return render(request,'admins/AdminViewUsers.html',{'data':data})


def Sendcropdata(request):
    data = cropyieldanalysismodel.objects.all()
    return render(request,'admins/AdminViewcropdetails.html',{'data':data})

def sendcrop(request):
    if request.method == 'GET':
        id = request.GET.get('id')
        print(' ID = ', id)
        loginid = request.session['loginid']
        cropyieldanalysismodel.objects.filter(id=id).update(status='sent')
        data = cropyieldanalysismodel.objects.filter(loginid=loginid)
        return render(request, 'admins/AdminViewcropdetails.html', {'data': data})




def storecsvdata(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        csvfile =TextIOWrapper( request.FILES['file'])
        columns = defaultdict(list)

        storecsvdata = csv.DictReader(csvfile)

        for row1 in storecsvdata:
                state = row1["state"]
                dist = row1["dist"]
                yeild = row1["yeild"]
                year = row1["year"]
                label = row1["label"]

                storedatamodel.objects.create(state=state, dist=dist, yeild=yeild,
                                                year=year, label=label)

        print("Name is ",csvfile)
        return HttpResponse('CSV file successful uploaded')
    else:

        return render(request, 'admins/storecropdata.html', {})


def MLprocess(request):
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
    accurancy = classifier.score(X_train, y_train)
    print("accurancy", accurancy)
    predicition =classification_report(y_test, y_pred)
    print("predicition",predicition)
    x = predicition.split()
    print("Toctal splits ", len(x))
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

    dict = {
        "accurancy": accurancy,
        #"predicition":predicition,
        'len0': x[0],
        'len1': x[1],
        'len2': x[2],
        'len3': x[3],
        'len4': x[4],
        'len5': x[5],
        'len6': x[6],
        'len7': x[7],
        'len8': x[8],
        'len9': x[9],
        'len10': x[10],
        'len11': x[11],
        'len12': x[12],
        'len13': x[13],
        'len14': x[14],
        'len15': x[15],
        'len16': x[16],
        'len17': x[17],
        'len18': x[18],
        'len19': x[19],
        'len20': x[20],
        'len21': x[21],
        'len22': x[22],
        'len23': x[23],
        'len24': x[24],
        'len25': x[25],
        'len26': x[26],
        'len27': x[27],
        'len28': x[28],

    }

    return render(request, 'admins/mlaccurancy.html',dict)