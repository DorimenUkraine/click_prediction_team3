import numpy as np
import pandas as pd
import category_encoders as ce
import math
import collections,csv
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import svm
from sklearn.pipeline import Pipeline


def drop_columns(cols,df):
    for col in cols:
        df.drop(columns=col,inplace=True)
    return df

def remove_time(df):
    df.drop(['request_received','local_time_of_request','time_recs_displayed'],axis=1,inplace=True)
    return df

def label_encode(df):
    df = df.apply(LabelEncoder().fit_transform)
    return df

def onehot_encode(df,cols):
    encoder=ce.OneHotEncoder(cols,return_df=1,drop_invariant=1,handle_missing='return_nan',use_cat_names=True)
    encoder.fit(X=df,y=df['set_clicked'])
    df=encoder.transform(df)
    return df

def catboost(df,col):
    encoder = ce.CatBoostEncoder(cols=[col],return_df=1,drop_invariant=1,handle_missing='return_nan',sigma=None,a=2)
    encoder.fit(X=df,y=df['set_clicked'])
    df=encoder.transform(df)
    return df

def catboost_multiple(df,cols):
    encoder = ce.CatBoostEncoder(cols,return_df=1,drop_invariant=1,handle_missing='return_nan',sigma=None,a=2)
    encoder.fit(X=df,y=df['set_clicked'])
    df = encoder.transform(df)
    return df

def catboost_multiple_test(df,testdf,cols):
    encoder = ce.CatBoostEncoder(cols,return_df=1,drop_invariant=1,handle_missing='return_nan',sigma=None,a=2)
    encoder.fit(X=df.drop(['set_clicked'], axis=1),y=df['set_clicked'])
    df=encoder.transform(df.drop(['set_clicked'], axis=1))
    testdf = encoder.transform(testdf)
    return df,testdf

def woencoder_train(df,cols):
    encoder=ce.WOEEncoder(cols,randomized=True,sigma=0.1)
    encoder.fit(X=df.iloc[:,:-1],y=df['set_clicked'])
    df = encoder.transform(df.iloc[:,:-1])
    return df

def woe_multiple_test(df,testdf,cols):
    encoder = ce.WOEEncoder(cols,randomized=True,sigma=0.1)
    encoder.fit(X=df.drop(['set_clicked'], axis=1),y=df['set_clicked'])
    df=encoder.transform(df.drop(['set_clicked'], axis=1))
    testdf = encoder.transform(testdf)
    return df,testdf

def group(df,cols,threshold):
    for col in cols:
        counts = df[col].value_counts()
        index = counts[counts<=threshold].index
        df[col] = df[col].replace(index,"other")
    return df


####### Configuration  ##################

test = False
catboost = True
woe = False
grid_search = False
decision_tree = True
svc = False
random_forest = False

########################################

# Read training data
lo={'recommendation_set_id':str, 'user_id':str, 'session_id':str, 'query_identifier':str,
'query_word_count':float, 'query_char_count':float, 'query_detected_language':str,
'query_document_id':str, 'document_language_provided':str, 'year_published':float,
'number_of_authors':float, 'abstract_word_count':float, 'abstract_char_count':float,
'abstract_detected_language':str, 'first_author_id':str,
'num_pubs_by_first_author':float, 'organization_id':str, 'application_type':str,
'item_type':str, 'request_received':str, 'hour_request_received':str,
'response_delivered':str, 'rec_processing_time':float, 'app_version':str, 'app_lang':str,
'user_os':str, 'user_os_version':str, 'user_java_version':str, 'user_timezone':str,
'country_by_ip':str, 'timezone_by_ip':str, 'local_time_of_request':str,
'local_hour_of_request':str, 'number_of_recs_in_set':float,
'recommendation_algorithm_id_used':str, 'algorithm_class':str, 'cbf_parser':str,
'search_title':str, 'search_keywords':str, 'search_abstract':str,
'time_recs_recieved':str, 'time_recs_displayed':str, 'time_recs_viewed':str,
'clicks':float, 'ctr':float,'set_clicked':float}
pars=['request_received', 'response_delivered','local_time_of_request','time_recs_recieved','time_recs_displayed','time_recs_viewed']

df=pd.read_csv('trainingdata.csv',na_values=["\\N","nA"], dtype=lo, parse_dates=pars)

# Clean the blog data
df=df[df.organization_id=='8']
# Drop columns not relevant for the blog
# response_delivered,number_of_recs_in_set,time_recs_recieved can only be used to filter data, not train.
# time_recs_viewed - important for filtering
cols_to_drop = ['recommendation_set_id','abstract_detected_language','user_id','user_java_version','user_os','user_os_version','year_published','session_id','query_document_id','ctr','response_delivered',
'app_version','app_lang','number_of_recs_in_set','time_recs_recieved','time_recs_viewed','document_language_provided','first_author_id','query_char_count','user_timezone','num_pubs_by_first_author','number_of_authors',
'organization_id','item_type','application_type','clicks','ctr','rec_processing_time']
df.drop(cols_to_drop,axis=1,inplace=True)
print(df.shape)


df = remove_time(df)

df['abstract_word_count'].fillna(df['abstract_word_count'].mode()[0], inplace=True)
df['abstract_char_count'].fillna(df['abstract_char_count'].mode()[0], inplace=True)
# algorithm_class
# cbf_parser
df['cbf_standard_QP']=df.cbf_parser.map(lambda x:1 if x=='standard_QP' else 0)
df['cbf_edismax_QP']=df.cbf_parser.map(lambda x:1 if x=='cbf_edismax_QP' else 0)
df['cbf_mlt_QP']=df.cbf_parser.map(lambda x:1 if x=='cbf_mlt_QP' else 0)
df['cbf_parser_used']=df.cbf_parser.map(lambda x: 1 if x else 0)
df.drop(columns=['cbf_parser'],inplace=True)

df.dropna(subset=['country_by_ip','query_word_count'],inplace=True)
df.fillna(method='ffill',inplace=True)

print(df.shape)

# Transform categorical data
y = df['set_clicked']  # Labels
if(test == False and catboost):
    df = catboost_multiple(df,['query_detected_language','query_identifier','country_by_ip','timezone_by_ip','algorithm_class','search_title','search_abstract','search_keywords'])
if(test == False and woe):
    df = woencoder_train(df,['query_detected_language','query_identifier','country_by_ip','timezone_by_ip','algorithm_class','search_title','search_abstract','search_keywords'])


#df = pd.get_dummies(data=df, drop_first=False)
#df = transform_categorical(df)

#print(df[df.isna().any(axis=1)])
#df = keep_known_columns(df)


print(df.shape)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(df.drop(['set_clicked'], axis=1), y, test_size=0.2,random_state=0) # 70% training and 30% test

if(grid_search):
    # Build the model, including the encoder
    model = Pipeline([
      ('encode_categorical', encoder),
      ('classifier', tree.DecisionTreeClassifier())
    ])
    # build a grid search
    grid = GridSearchCV(model, param_grid=params, cv=5).fit(X_train, y_train)
    preds_class = grid.predict(X_test)
    acc=sum(y_test==preds_class)/len(y_test)
    print(set(preds_class) - set(preds_class))
    #print(i)
    print("accuracy: ",acc)
    print("precision: ",metrics.precision_score(y_test, preds_class))
    print("f1: ",metrics.f1_score(y_test, preds_class))
    print("cm:",metrics.confusion_matrix(y_test, preds_class))

if(decision_tree):
    for i in range(2,12,1):
        clf = tree.DecisionTreeClassifier(max_depth=i,).fit(X_train, y_train)

        preds_class = clf.predict(X_test)
        acc=sum(y_test==preds_class)/len(y_test)
        print(set(preds_class) - set(preds_class))
        print(i)
        print("accuracy: ",acc)
        print("precision: ",metrics.precision_score(y_test, preds_class))
        print("f1: ",metrics.f1_score(y_test, preds_class))
        print("cm:",metrics.confusion_matrix(y_test, preds_class))


if(svc):
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)
    preds_class = clf.predict(X_test)
    acc=sum(y_test==preds_class)/len(y_test)
    print("accuracy: ",acc)
    print("precision: ",metrics.precision_score(y_test, preds_class))
    print("f1: ",metrics.f1_score(y_test, preds_class))
    print("cm:",metrics.confusion_matrix(y_test, preds_class))

    #print("kaggle: ",sum(clf.predict(kag_del)))

    #print("kaggle: ",sum(clf.predict(kag_del)))
if(random_forest):
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(X_train, y_train)
    preds_class = clf.predict(X_test)
    acc=sum(y_test==preds_class)/len(y_test)
    print("accuracy: ",acc)
    print("f1: ",metrics.f1_score(y_test, preds_class))
    print("cm:",metrics.confusion_matrix(y_test, preds_class))

if(test):
    testdf = pd.read_csv('testingdata.csv',na_values=["\\N","nA"], dtype=lo, parse_dates=pars)
    print("read test data")
    # Clean the blog data
    testdf=testdf[testdf.organization_id=='8']
    print(testdf.shape)
    # Drop columns not relevant for the blog
    # response_delivered,number_of_recs_in_set,time_recs_recieved can only be used to filter data, not train.
    # time_recs_viewed - important for filtering
    set_id = testdf['recommendation_set_id']
    cols_to_drop = ['recommendation_set_id','abstract_detected_language','user_id','user_java_version','user_os','user_os_version','year_published','session_id','query_document_id','ctr','response_delivered',
    'app_version','app_lang','number_of_recs_in_set','time_recs_recieved','time_recs_viewed','document_language_provided','first_author_id','query_char_count','user_timezone','num_pubs_by_first_author','number_of_authors',
    'organization_id','item_type','application_type','clicks','ctr','rec_processing_time','set_clicked']
    testdf.drop(cols_to_drop,axis=1,inplace=True)
    testdf = remove_time(testdf)

    testdf['abstract_word_count'].fillna(testdf['abstract_word_count'].mode()[0], inplace=True)
    testdf['abstract_char_count'].fillna(testdf['abstract_char_count'].mode()[0], inplace=True)
    testdf['query_word_count'].fillna(testdf['query_word_count'].mode()[0], inplace=True)
    testdf['country_by_ip'].fillna(testdf['country_by_ip'].mode()[0], inplace=True)
     # algorithm_class
    # cbf_parser
    testdf['cbf_standard_QP']=testdf.cbf_parser.map(lambda x:1 if x=='standard_QP' else 0)
    testdf['cbf_edismax_QP']=testdf.cbf_parser.map(lambda x:1 if x=='cbf_edismax_QP' else 0)
    testdf['cbf_mlt_QP']=testdf.cbf_parser.map(lambda x:1 if x=='cbf_mlt_QP' else 0)
    testdf['cbf_parser_used']=testdf.cbf_parser.map(lambda x: 1 if x else 0)
    testdf.drop(columns=['cbf_parser'],inplace=True)



    testdf.fillna(method='ffill',inplace=True)
    print(testdf[testdf.isna().any(axis=1)])

    print(testdf.shape)
    # Transform categorical data
    new_y = df['set_clicked']
    df,testdf = woe_multiple_test(df,testdf,['query_detected_language','query_identifier','country_by_ip','timezone_by_ip','algorithm_class','search_title','search_abstract','search_keywords'])
    print("Train shape:",df.shape)
    print("Test shape: ",testdf.shape)
    #testdf.drop(['set_clicked'], axis=1,inplace=True)

    print("Training...")
    clf = tree.DecisionTreeClassifier(max_depth=12).fit(df,new_y)

    # Get missing columns in the training test
    #missing_cols = set(X.columns ) - set(testdf.columns )
    # Add a missing column in test set with default value equal to 0
    #for c in missing_cols:
        #testdf[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    #testdf = testdf[X.columns]


    pred = clf.predict(testdf)
    print(testdf.shape)
    a=pd.DataFrame.from_dict({
        'recommendation_set_id' : set_id,
        'set_clicked': pred,
    })
    print("Writing to csv..")
    a.to_csv(r'attempt_woe_depth12.csv',index=False)
