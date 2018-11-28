import keras
from keras.models import load_model
from flask import Flask,Response,send_from_directory,redirect
import pickle
from datetime import datetime
from flask import json
from flask import request
from flask import render_template
import os
import numpy as np
import requests
import re

with open("tfidf-feature.pickle","rb") as file:
    (tfidf_vectorizer,tfidf) = pickle.load(file)

# load tf_feature_dict
with open("tf_feature_dict.pickle","rb") as file:
    (tf_feature_dict,y_train_label_l) = pickle.load(file)

with open("embed_weights.pickle","rb") as file:
    (tf_embed_dict,embed_weights) = pickle.load(file)

# subcategories label
with open("label-subcats.pickle","rb") as file:
    y_train_label_subcats = pickle.load(file)

abstract = 'As large-scale dense and often randomly deployed wireless sensor networks\n(WSNs) become widespread, local information exchange between co-located sets of\nnodes may play a significant role in handling the excessive traffic volume.\nMoreover, to account for the limited life-span of the wireless devices,\nharvesting the energy of the network transmissions provides significant\nbenefits to the lifetime of such networks. In this paper, we study the\nperformance of communication in dense networks with wireless energy harvesting\n(WEH)-enabled sensor nodes. In particular, we examine two different\ncommunication scenarios (direct and cooperative) for data exchange and we\nprovide theoretical expressions for the probability of successful\ncommunication. Then, considering the importance of lifetime in WSNs, we employ\nstate-of-the-art WEH techniques and realistic energy converters, quantifying\nthe potential energy gains that can be achieved in the network. Our analytical\nderivations, which are validated by extensive Monte-Carlo simulations,\nhighlight the importance of WEH in dense networks and identify the trade-offs\nbetween the direct and cooperative communication scenarios.'


def remove_symbol(my_str: str):
    reg_exclusion = r'[^a-z| ]'
    return re.sub(reg_exclusion, ' ', my_str)

def alphabetic_only_l(abstract_sent):
    abstract_sent = abstract_sent.lower()
    clean_sent = remove_symbol(abstract_sent)
    list_sent = []
    for tokens in clean_sent.split(" "):
        # remove all tokens that are not alphabetic
        if len(tokens) > 0:
            list_sent.append(tokens)
    return list_sent

def list_to_feature(abstract_l,feature_dict,max=150):
    abstract_mat = np.zeros((1, max), dtype=np.int)
    feat_x = []
    i = 0
    for my_x in abstract_l:
        try:
            feat_x.append(feature_dict[my_x])
            i += 1
        except:
            pass
        if i > 150:
            break
    #print(feat_x)
    #print(len(feat_x))
    abstract_mat[0, -len(feat_x):] = feat_x
    #print(abstract_mat)
    return abstract_mat

abstract_l = alphabetic_only_l(abstract)
#print(abstract_l)

# load tfidf dense model
tfidf_model = load_model("nn-model-drop-multilabel-b9.h5")
# load convolutional model
conv_model = load_model("nn-model-multilabel-conv1c-9.h5")
# load lstm model
lstm_model = load_model("nn-model-lstm-embed-multilabel-c9.h5")

# sub categories model
lstm_subcat_model = load_model("nn-model-lstm-embed-multilabel-subcats-d99.h5")


tfidf_input = tfidf_vectorizer.transform([" ".join(abstract_l)])
#print(tfidf_input)

def predict_abstract(abstract_l):
    tfidf_input = tfidf_vectorizer.transform([" ".join(abstract_l)])
    tfidf_pred = tfidf_model.predict(tfidf_input)
    tfidf_prob = np.sort(tfidf_pred[0])[::-1]
    tfidf_index = np.argsort(tfidf_pred[0])[::-1]
    tfidf_class = np.array(y_train_label_l)[np.argsort(tfidf_pred[0])[::-1]]
    tfidf_result = list(zip(tfidf_class.tolist(), tfidf_index.tolist(), tfidf_prob.tolist()))

    abstract_mat = list_to_feature(abstract_l, tf_feature_dict)

    conv_pred = conv_model.predict(abstract_mat)
    conv_prob = np.sort(conv_pred[0])[::-1]
    conv_index = np.argsort(conv_pred[0])[::-1]
    conv_class = np.array(y_train_label_l)[np.argsort(conv_pred[0])[::-1]]
    conv_result = list(zip(conv_class.tolist(), conv_index.tolist(), conv_prob.tolist()))

    abstract_mat = list_to_feature(abstract_l, tf_embed_dict)

    lstm_pred = lstm_model.predict(abstract_mat)
    lstm_prob = np.sort(lstm_pred[0])[::-1]
    lstm_index = np.argsort(lstm_pred[0])[::-1]
    lstm_class = np.array(y_train_label_l)[np.argsort(lstm_pred[0])[::-1]]
    lstm_result = list(zip(lstm_class.tolist(), lstm_index.tolist(), lstm_prob.tolist()))

    # subcategories model
    lstm_s_pred = lstm_subcat_model.predict(abstract_mat)
    lstm_s_prob = np.sort(lstm_s_pred[0])[::-1]
    lstm_s_index = np.argsort(lstm_s_pred[0])[::-1]
    lstm_s_class = np.array(y_train_label_subcats)[np.argsort(lstm_s_pred[0])[::-1]]
    lstm_s_result = list(zip(lstm_s_class.tolist(), lstm_s_index.tolist(), lstm_s_prob.tolist()))

    return (tfidf_result,conv_result,lstm_result,lstm_s_result)

'''
max = 150
abstract_mat = np.zeros((1,max),dtype=np.int)
feat_x = []
i = 0
for my_x in abstract_l:
    try:
        feat_x.append(tf_feature_dict[my_x])
        i+=1
    except:
        pass
    if i>150:
        break

print(feat_x)
print(len(feat_x))
abstract_mat[0,-len(feat_x):] = feat_x
print(abstract_mat)

# load model
conv_pred = conv_model.predict(abstract_mat)
conv_prob = np.sort(conv_pred[0])[::-1]
conv_index = np.argsort(conv_pred[0])[::-1]
conv_class = np.array(y_train_label_l)[np.argsort(conv_pred[0])[::-1]]
print(list(zip(conv_class,conv_index,conv_prob)))



max = 150
abstract_mat = np.zeros((1, max), dtype=np.int)
feat_x = []
i = 0
for my_x in abstract_l:
    try:
        feat_x.append(tf_embed_dict[my_x])
        i += 1
    except:
        pass
    if i > 150:
        break

print(feat_x)
print(len(feat_x))
abstract_mat[0, -len(feat_x):] = feat_x
print(abstract_mat)

lstm_pred = lstm_model.predict(abstract_mat)
lstm_prob = np.sort(lstm_pred[0])[::-1]
lstm_index = np.argsort(lstm_pred[0])[::-1]
lstm_class = np.array(y_train_label_l)[np.argsort(lstm_pred[0])[::-1]]
print(list(zip(lstm_class,lstm_index,lstm_prob)))
'''

app = Flask(__name__)


@app.route('/category_pred', methods=['POST'])
def category_pred():  # pragma: no cover
    """

    :return:
    """
    result = {"status": 1, "message": "not enough information to predict", "prediction": None}
    if request.method == "POST":
        if request.headers['Content-Type'] == 'application/json':
            parameters = request.json
            print(parameters)
            abstract = parameters["abstract"]

            abstract_l = alphabetic_only_l(abstract)
            (tfidf, conv,lstm) = predict_abstract(abstract_l)
            result["prediction"] = {"tfidf": tfidf, "conv": conv,"lstm": lstm}
            result["status"] = 0
            result["message"] = "OK"

    return json.dumps(result)

@app.route('/', methods=['GET','POST'])
def web_root():  # pragma: no cover
    """

    :return:
    """
    print(request.headers)
    print(request.method)
    #if request.headers['Content-Type'] == 'application/json':
    submitted = False
    if request.method == "POST":
        if request.headers['Content-Type'] == 'application/x-www-form-urlencoded':
            my_data = request.form
            #print(my_data)
            abstract = my_data['abstract_input']
            submitted = True
    else:
        abstract = 'As large-scale dense and often randomly deployed wireless sensor networks\n(WSNs) become widespread, local information exchange between co-located sets of\nnodes may play a significant role in handling the excessive traffic volume.\nMoreover, to account for the limited life-span of the wireless devices,\nharvesting the energy of the network transmissions provides significant\nbenefits to the lifetime of such networks. In this paper, we study the\nperformance of communication in dense networks with wireless energy harvesting\n(WEH)-enabled sensor nodes. In particular, we examine two different\ncommunication scenarios (direct and cooperative) for data exchange and we\nprovide theoretical expressions for the probability of successful\ncommunication. Then, considering the importance of lifetime in WSNs, we employ\nstate-of-the-art WEH techniques and realistic energy converters, quantifying\nthe potential energy gains that can be achieved in the network. Our analytical\nderivations, which are validated by extensive Monte-Carlo simulations,\nhighlight the importance of WEH in dense networks and identify the trade-offs\nbetween the direct and cooperative communication scenarios.'

    html_all = ""

    html_all+= '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>arXiv Predictor</title>
    </head>
    <body>
        <form target="_self" method="post">
        <table>
            <tr><td colspan="3">Abstract:<br /><textarea name="abstract_input" id="abstract_input" rows="8" cols="100">'''
    html_all+=abstract
    html_all+='''</textarea></td></tr>
            <tr><td colspan="3"><button name="predict_cat" type="submit">Predict</button></td></tr>
        </table>
        </form>            
    '''

    if submitted:
        abstract_l = alphabetic_only_l(abstract)
        (tfidf, conv,lstm,lstm_s) = predict_abstract(abstract_l)
        #print(tfidf)
        #print(conv)
        #print(lstm)
        html_all+='''
                <table border="1">
                <tr>
                    <td colspan=2>TF-IDF NN</td>
                    <td colspan=2>Convolutional NN</td>
                    <td colspan=2>LSTM NN</td>
                </tr>
                <tr>
                    <td>Category</td><td>Probability</td>
                    <td>Category</td><td>Probability</td>
                    <td>Category</td><td>Probability</td>
                </tr>
        '''
        for x in range(5):
            html_all += '''
                            <tr>
                                <td>'''+str(tfidf[x][0])+'''</td><td>'''+str(tfidf[x][2])+'''</td>
                                <td>'''+str(conv[x][0])+'''</td><td>'''+str(conv[x][2])+'''</td>
                                <td>'''+str(lstm[x][0])+'''</td><td>'''+str(lstm[x][2])+'''</td>
                            </tr>
                    '''
        html_all+='''
            </table>
            <table border="1">
                <tr>
                    <td colspan=2>LSTM NN for SubCategories</td>
                </tr>
                <tr>
                    <td>Category</td><td>Probability</td>
                </tr>'''
        for x in range(10):
            html_all += '''
                            <tr>
                                <td>'''+str(lstm_s[x][0])+'''</td><td>'''+str(lstm_s[x][2])+'''</td>
                            </tr>
                    '''
        html_all+='''
            </table>            
            '''
    html_all+='''            
    </body>
    </html>
    '''

    return html_all


if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0")


'''
tfidf_pred = tfidf_model.predict(tfidf_input)
tfidf_prob = np.sort(tfidf_pred[0])[::-1]
tfidf_index = np.argsort(tfidf_pred[0])[::-1]
tfidf_class = np.array(y_train_label_l)[np.argsort(tfidf_pred[0])[::-1]]
print(list(zip(tfidf_class,tfidf_index,tfidf_prob)))

"""
max = 150
abstract_mat = np.zeros((1,max),dtype=np.int)
feat_x = []
i = 0
for my_x in abstract_l:
    try:
        feat_x.append(tf_feature_dict[my_x])
        i+=1
    except:
        pass
    if i>150:
        break

print(feat_x)
print(len(feat_x))
abstract_mat[0,-len(feat_x):] = feat_x
print(abstract_mat)
"""
abstract_mat = list_to_feature(abstract_l,tf_feature_dict)

# load model
conv_pred = conv_model.predict(abstract_mat)
conv_prob = np.sort(conv_pred[0])[::-1]
conv_index = np.argsort(conv_pred[0])[::-1]
conv_class = np.array(y_train_label_l)[np.argsort(conv_pred[0])[::-1]]
print(list(zip(conv_class,conv_index,conv_prob)))


with open("embed_weights.pickle","rb") as file:
    (tf_embed_dict,embed_weights) = pickle.load(file)

"""
max = 150
abstract_mat = np.zeros((1, max), dtype=np.int)
feat_x = []
i = 0
for my_x in abstract_l:
    try:
        feat_x.append(tf_embed_dict[my_x])
        i += 1
    except:
        pass
    if i > 150:
        break

print(feat_x)
print(len(feat_x))
abstract_mat[0, -len(feat_x):] = feat_x
print(abstract_mat)
"""
abstract_mat = list_to_feature(abstract_l,tf_embed_dict)

lstm_pred = lstm_model.predict(abstract_mat)
lstm_prob = np.sort(lstm_pred[0])[::-1]
lstm_index = np.argsort(lstm_pred[0])[::-1]
lstm_class = np.array(y_train_label_l)[np.argsort(lstm_pred[0])[::-1]]
print(list(zip(lstm_class,lstm_index,lstm_prob)))
'''

'''
def extract_time(time_str,time_format="%Y-%m-%d %H:%M:%S UTC"):
    datetime_object = datetime.strptime(time_str, time_format)
    return (datetime_object.year,datetime_object.month,datetime_object.day,datetime_object.weekday(),datetime_object.hour)

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)

app = Flask(__name__)

# load model
with open("all_model.pickle","rb") as file:
    (est2, est2b, est2c, regr_1d, regr_2d) = pickle.load(file)

@app.route('/', methods=['GET'])
def web_root():  # pragma: no cover
    return redirect("/taxi/index.html", code=302)

@app.route('/taxi/', methods=['GET'])
def taxi_web():  # pragma: no cover
    return redirect("/taxi/index.html", code=302)

@app.route('/taxi/<path:path>')
def send_js(path):
    return send_from_directory('taxi_web', path)

@app.route("/calculate_fare",methods=["POST"])
def calculate_fare():
    if request.headers['Content-Type'] == 'application/json':
        parameters = request.json
        print(parameters)
        start_loc = parameters["start"]
        dest_loc = parameters["dest"]
        pickup_time = parameters["pickup_time"]
        pass_count = parameters["passenger_count"]

        my_time = extract_time(pickup_time,"%Y/%m/%d %H:%M")

        # calculate distance using gogole api
        dist_url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins="+str(start_loc["lat"])+","+str(start_loc["lng"])+"&destinations="+str(dest_loc["lat"])+","+str(dest_loc["lng"])+"&key=AIzaSyA7_KvLPtVesTg7NLxXnL_czMlX7GcyXXw";
        r = requests.get(url=dist_url)

        # extracting data in json format
        data = r.json()

        print(data)
        dist_mile = float(data["rows"][0]["elements"][0]["distance"]["text"].replace(" mi",""))
        dist_text = data["rows"][0]["elements"][0]["distance"]["text"]
        start_place = data["origin_addresses"][0]
        dest_place = data["destination_addresses"][0]
        duration = data["rows"][0]["elements"][0]["duration"]["text"]
        #print(dist_mile)
        """         
        pickup_latitude
        pickup_longitude
        dropoff_latitude
        dropoff_longitude
        passenger_count
        distance
        year
        month
        dayofweek
        hour
        """

        max_lat = 40.9
        min_lat = 40.6
        max_lon = -73.7
        min_lon = -74.1
        type = 0

        # check if the location between NY City Range
        # make prediction using location as well
        if ((start_loc["lat"]>min_lat)&
           (start_loc["lat"]<max_lat)&
           (dest_loc["lat"]>min_lat)&
           (dest_loc["lat"]<max_lat)&
           (start_loc["lng"]>min_lon)&
           (start_loc["lng"]<max_lon)&
           (dest_loc["lng"]>min_lon)&
           (dest_loc["lng"]<max_lon)):
            my_X = np.zeros((1,10));
            my_X[0,0] = start_loc["lat"]
            my_X[0,1] = start_loc["lng"]
            my_X[0,2] = dest_loc["lat"]
            my_X[0,3] = dest_loc["lng"]
            my_X[0,4] = int(pass_count)
            my_X[0,5] = dist_mile
            my_X[0,6] = my_time[0]
            my_X[0,7] = my_time[1]
            my_X[0,8] = my_time[3]
            my_X[0,9] = my_time[4]

            #my_xc = sm.add_constant(my_X)

            fare_pred = regr_1d.predict(my_X)

        else:
            type=1
            my_X = np.zeros((2,6))
            my_X[0,0] = int(pass_count)
            my_X[0,1] = dist_mile
            my_X[0,2] = my_time[0]
            my_X[0,3] = my_time[1]
            my_X[0,4] = my_time[3]
            my_X[0,5] = my_time[4]

            my_xc = sm.add_constant(my_X)
            print(my_xc)
            print(my_xc.shape)

            fare_pred = est2c.predict(my_xc)

        print(fare_pred)

        #distance_api =

        #return "data:image/png;base64,{}".format(image_encoded)
        #return fare
        return json.dumps({"status": 0,"type":type,"start":start_place,"dest":dest_place,"distance":dist_text,"duration":duration,"fare":fare_pred.tolist()[0]})
    return json.dumps({"status": 1})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
'''