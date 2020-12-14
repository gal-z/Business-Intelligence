import pandas as pd
import re
import datetime
from sklearn.preprocessing import OneHotEncoder



data_apps = pd.read_csv("Adults.csv", encoding="ISO-8859-1")


# def change_coll_only_num(coll):
#     app_size = []
#     for i in coll:
#        i=re.sub('M','000000',i)
#        i = re.sub('k', '000', i)
#        i=re.sub("Varies with device","0",i)
#        if int((float(i)))!=float(i):
#         i= float(i)*1000000000
#        i=int(float(i))
#        if i <10:
#            i=i*1000000000
#        app_size.append(i)
#     coll=app_size
#     return coll


def clean_install_native_country(install_coull):
    data_apps_download = install_coull.str.replace('?', 'United-States')
    # data_apps_download_new = data_apps_download.str.replace(',', '').astype(int)
    return data_apps_download

def clean_install_occupation(install_coull):
    data_apps_download = install_coull.str.replace('?', 'Prof-specialty')
    # data_apps_download_new = data_apps_download.str.replace(',', '').astype(int)
    return data_apps_download

def clean_install_workclass(install_coull):
    data_apps_download = install_coull.str.replace('?', 'Private')
    # data_apps_download_new = data_apps_download.str.replace(',', '').astype(int)
    return data_apps_download

def clean_install_income(install_coull):
    data_apps_download = install_coull.str.replace('<=50K', '0')
    data_apps_download_new = data_apps_download.str.replace('>50K', '1').astype(int)
    return data_apps_download_new

def clean_install_sex(install_coull):
    data_apps_download = install_coull.str.replace('Female', '0')
    data_apps_download_new = data_apps_download.str.replace('Male', '1').astype(int)
    return data_apps_download_new

def find_app_value(x, dict):
        return dict[x]

def change_country_to_num(coll):
    apps_num_dict = {}
    apps_name_no_dupli = set(coll)
    num = 1
    for i in apps_name_no_dupli:
        if i not in apps_num_dict.keys():
            apps_num_dict[i] = int(num)
            num += 1

    data_clean=coll.apply(lambda x: find_app_value(str(x),apps_num_dict))
    print(apps_num_dict)
    return data_clean

# def change_time (date_index):
#     date_time_str = date_index
#     date_time_obj = datetime.datetime.strptime(date_time_str, '%B %d, %Y')
#     return date_time_obj
#
#
# def change_app_coll_nan(data_coll):
#     pta = pd.Series(data_coll)
#     pta[pta.isnull()]= 'has_no_name'
#     return pta
#
# def change_rating_coll_nan(data_coll):
#     pto = pd.Series(data_coll)
#     pto[pto.isnull()]=0
#     return pto
#
# def clean_nan_from_sentiment_subj(data):
#     pta = pd.Series(data)
#     pta[pta.isnull()]= pta.mean()
#     return pta
#
#
# def clean_nan_from_sentiment_pol(data):
#     ptb = pd.Series(data)
#     ptb[ptb.isnull()]= ptb.mean()
#     return ptb
#
# def insert_most_frequent_to_nan_sentiment(data):
#     ptc = pd.Series(data)
#     mode = ptc.mode()
#     ptc[ptc.isnull()]= 'Positive'
#     return ptc


print(list(data_apps.head(0)))

data_apps['native.country']= clean_install_native_country(data_apps['native.country'])
print( 'most common occupation is',data_apps['occupation'].mode())

data_apps['occupation']= clean_install_occupation(data_apps['occupation'])

print('most common workclass is',data_apps['workclass'].mode())

data_apps['workclass']= clean_install_workclass(data_apps['workclass'])
#_____________________________data cleen finished___________________________________________________________________________________
data_apps['income']= clean_install_income(data_apps['income'])
print(data_apps['income'])

data_apps['sex']= clean_install_sex(data_apps['sex'])
print(data_apps['sex'])
#____________making string columns to nums_______________________________________________________________________
data_apps['native.country']= change_country_to_num(data_apps['native.country'])
print(data_apps['native.country'])

data_apps['relationship']= change_country_to_num(data_apps['relationship'])
print(data_apps['relationship'])

data_apps['occupation']= change_country_to_num(data_apps['occupation'])
print(data_apps['occupation'])

data_apps['marital.status']= change_country_to_num(data_apps['marital.status'])
print(data_apps['marital.status'])

data_apps['education']= change_country_to_num(data_apps['education'])
print(data_apps['education'])

data_apps['workclass']= change_country_to_num(data_apps['workclass'])
print(data_apps['workclass'])


#______________making 'race' one hot vector______________________________
y=pd.Series(data_apps['race'])
x = pd.get_dummies(y)
data_apps= pd.concat([data_apps,x],axis=1)
print(data_apps.head(0))
data_apps=data_apps.drop(['race'],axis=1)
print(data_apps.head(0))

print(data_apps.head())



