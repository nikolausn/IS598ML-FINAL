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

# class definition
cat_dict = {
'astro-ph':'Astrophysics',
'cond-mat':'Condensed Matter',
'cs':'Computer Science',
'econ':'Economics',
'eess':'Electrical Engineering and System Science',
'gr-qc':'General Relativity and Quantum Cosmology',
'hep-ex':'High Energy Physics - Experiment',
'hep-lat':'High Energy Physics - Lattice',
'hep-ph':'High Energy Physics - Phenomenology',
'hep-th':'High Energy Physics - Theory',
'math':'Mathematics',
'math-ph':'Mathematical Physics',
'nlin':'Nonlinear Sciences',
'nucl-ex':'Nuclear Experiment',
'nucl-th':'Nuclear Theory',
'physics':'Physics',
'q-bio':'Quantitative Biology',
'q-fin':'Quantitative Finance',
'quant-ph':'Quantum Physics',
'stat':'Statistics'
}

sub_cat_dict = {
'astro-ph':'Astrophysics',
'astro-ph.CO':'Cosmology and Nongalactic Astrophysics',
'astro-ph.EP':'Earth and Planetary Astrophysics',
'astro-ph.GA':'Astrophysics of Galaxies',
'astro-ph.HE':'High Energy Astrophysical Phenomena',
'astro-ph.IM':'Instrumentation and Methods for Astrophysics',
'astro-ph.SR':'Solar and Stellar Astrophysics',
'cond-mat.dis-nn':'Disordered Systems and Neural Networks',
'cond-mat.mes-hall':'Mesoscale and Nanoscale Physics',
'cond-mat.mtrl-sci':'Materials Science',
'cond-mat.other':'Other Condensed Matter',
'cond-mat.quant-gas':'Quantum Gases',
'cond-mat.soft':'Soft Condensed Matter',
'cond-mat.stat-mech':'Statistical Mechanics',
'cond-mat.str-el':'Strongly Correlated Electrons',
'cond-mat.supr-con':'Superconductivity',
'cs.AI':'Artificial Intelligence',
'cs.AR':'Hardware Architecture',
'cs.CC':'Computational Complexity',
'cs.CE':'Computational Engineering, Finance, and Science',
'cs.CG':'Computational Geometry',
'cs.CL':'Computation and Language',
'cs.CR':'Cryptography and Security',
'cs.CV':'Computer Vision and Pattern Recognition',
'cs.CY':'Computers and Society',
'cs.DB':'Databases',
'cs.DC':'Distributed, Parallel, and Cluster Computing',
'cs.DL':'Digital Libraries',
'cs.DM':'Discrete Mathematics',
'cs.DS':'Data Structures and Algorithms',
'cs.ET':'Emerging Technologies',
'cs.FL':'Formal Languages and Automata Theory',
'cs.GL':'General Literature',
'cs.GR':'Graphics',
'cs.GT':'Computer Science and Game Theory',
'cs.HC':'Human-Computer Interaction',
'cs.IR':'Information Retrieval',
'cs.IT':'Information Theory',
'cs.LG':'Machine Learning',
'cs.LO':'Logic in Computer Science',
'cs.MA':'Multiagent Systems',
'cs.MM':'Multimedia',
'cs.MS':'Mathematical Software',
'cs.NA':'Numerical Analysis',
'cs.NE':'Neural and Evolutionary Computing',
'cs.NI':'Networking and Internet Architecture',
'cs.OH':'Other Computer Science',
'cs.OS':'Operating Systems',
'cs.PF':'Performance',
'cs.PL':'Programming Languages',
'cs.RO':'Robotics',
'cs.SC':'Symbolic Computation',
'cs.SD':'Sound',
'cs.SE':'Software Engineering',
'cs.SI':'Social and Information Networks',
'cs.SY':'Systems and Control',
'econ.EM':'Econometrics',
'eess.AS':'Audio and Speech Processing',
'eess.IV':'Image and Video Processing',
'eess.SP':'Signal Processing',
'gr-qc':'General Relativity and Quantum Cosmology',
'hep-ex':'High Energy Physics - Experiment',
'hep-lat':'High Energy Physics - Lattice',
'hep-ph':'High Energy Physics - Phenomenology',
'hep-th':'High Energy Physics - Theory',
'math.AC':'Commutative Algebra',
'math.AG':'Algebraic Geometry',
'math.AP':'Analysis of PDEs',
'math.AT':'Algebraic Topology',
'math.CA':'Classical Analysis and ODEs',
'math.CO':'Combinatorics',
'math.CT':'Category Theory',
'math.CV':'Complex Variables',
'math.DG':'Differential Geometry',
'math.DS':'Dynamical Systems',
'math.FA':'Functional Analysis',
'math.GM':'General Mathematics',
'math.GN':'General Topology',
'math.GR':'Group Theory',
'math.GT':'Geometric Topology',
'math.HO':'History and Overview',
'math.IT':'Information Theory',
'math.KT':'K-Theory and Homology',
'math.LO':'Logic',
'math.MG':'Metric Geometry',
'math.MP':'Mathematical Physics',
'math.NA':'Numerical Analysis',
'math.NT':'Number Theory',
'math.OA':'Operator Algebras',
'math.OC':'Optimization and Control',
'math.PR':'Probability',
'math.QA':'Quantum Algebra',
'math.RA':'Rings and Algebras',
'math.RT':'Representation Theory',
'math.SG':'Symplectic Geometry',
'math.SP':'Spectral Theory',
'math.ST':'Statistics Theory',
'math-ph':'Mathematical Physics',
'nlin.AO':'Adaptation and Self-Organizing Systems',
'nlin.CD':'Chaotic Dynamics',
'nlin.CG':'Cellular Automata and Lattice Gases',
'nlin.PS':'Pattern Formation and Solitons',
'nlin.SI':'Exactly Solvable and Integrable Systems',
'nucl-ex':'Nuclear Experiment',
'nucl-th':'Nuclear Theory',
'physics.acc-ph':'Accelerator Physics',
'physics.ao-ph':'Atmospheric and Oceanic Physics',
'physics.app-ph':'Applied Physics',
'physics.atm-clus':'Atomic and Molecular Clusters',
'physics.atom-ph':'Atomic Physics',
'physics.bio-ph':'Biological Physics',
'physics.chem-ph':'Chemical Physics',
'physics.class-ph':'Classical Physics',
'physics.comp-ph':'Computational Physics',
'physics.data-an':'Data Analysis, Statistics and Probability',
'physics.ed-ph':'Physics Education',
'physics.flu-dyn':'Fluid Dynamics',
'physics.gen-ph':'General Physics',
'physics.geo-ph':'Geophysics',
'physics.hist-ph':'History and Philosophy of Physics',
'physics.ins-det':'Instrumentation and Detectors',
'physics.med-ph':'Medical Physics',
'physics.optics':'Optics',
'physics.plasm-ph':'Plasma Physics',
'physics.pop-ph':'Popular Physics',
'physics.soc-ph':'Physics and Society',
'physics.space-ph':'Space Physics',
'q-bio.BM':'Biomolecules',
'q-bio.CB':'Cell Behavior',
'q-bio.GN':'Genomics',
'q-bio.MN':'Molecular Networks',
'q-bio.NC':'Neurons and Cognition',
'q-bio.OT':'Other Quantitative Biology',
'q-bio.PE':'Populations and Evolution',
'q-bio.QM':'Quantitative Methods',
'q-bio.SC':'Subcellular Processes',
'q-bio.TO':'Tissues and Organs',
'q-fin.CP':'Computational Finance',
'q-fin.EC':'Economics',
'q-fin.GN':'General Finance',
'q-fin.MF':'Mathematical Finance',
'q-fin.PM':'Portfolio Management',
'q-fin.PR':'Pricing of Securities',
'q-fin.RM':'Risk Management',
'q-fin.ST':'Statistical Finance',
'q-fin.TR':'Trading and Market Microstructure',
'quant-ph':'Quantum Physics',
'stat.AP':'Applications',
'stat.CO':'Computation',
'stat.ME':'Methodology',
'stat.ML':'Machine Learning',
'stat.OT':'Other Statistics',
'stat.TH':'Statistics Theory'
}

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
# to much consuming memory
#tfidf_model = load_model("nn-model-drop-multilabel-b9.h5")
# load convolutional model
conv_model = load_model("nn-model-multilabel-conv1c-9.h5")
# load lstm model
lstm_model = load_model("nn-model-lstm-embed-multilabel-c9.h5")

# sub categories model
lstm_subcat_model = load_model("nn-model-lstm-embed-multilabel-subcats-d99.h5")


tfidf_input = tfidf_vectorizer.transform([" ".join(abstract_l)])
#print(tfidf_input)

def predict_abstract(abstract_l):
    '''
    tfidf_input = tfidf_vectorizer.transform([" ".join(abstract_l)])
    tfidf_pred = tfidf_model.predict(tfidf_input)
    tfidf_prob = np.sort(tfidf_pred[0])[::-1]
    tfidf_index = np.argsort(tfidf_pred[0])[::-1]
    tfidf_class = np.array(y_train_label_l)[np.argsort(tfidf_pred[0])[::-1]]
    tfidf_result = list(zip(tfidf_class.tolist(), tfidf_index.tolist(), tfidf_prob.tolist()))
    '''

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

    #return (tfidf_result,conv_result,lstm_result,lstm_s_result)
    return (conv_result,lstm_result,lstm_s_result)

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
            #(tfidf, conv,lstm) = predict_abstract(abstract_l)
            (conv,lstm,lstm_s) = predict_abstract(abstract_l)
            #result["prediction"] = {"tfidf": tfidf, "conv": conv,"lstm": lstm}
            result["prediction"] = {"conv": conv,"lstm": lstm}
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
        (conv,lstm,lstm_s) = predict_abstract(abstract_l)
        #print(tfidf)
        #print(conv)
        #print(lstm)
        html_all+='''
                <table border="1">
                <tr>
                    <td colspan=2>Convolutional NN</td>
                    <td colspan=2>LSTM NN</td>
                </tr>
                <tr>
                    <td>Category</td><td>Probability</td>
                    <td>Category</td><td>Probability</td>
                </tr>
        '''
        for x in range(5):
            html_all += '''
                            <tr>
                                <td>'''+"{} ({})".format(cat_dict[conv[x][0]],conv[x][0])+'''</td><td>'''+str(conv[x][2])+'''</td>
                                <td>'''+"{} ({})".format(cat_dict[lstm[x][0]],lstm[x][0])+'''</td><td>'''+str(lstm[x][2])+'''</td>
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
                                <td>'''+"{} ({})".format(sub_cat_dict[lstm_s[x][0]],lstm_s[x][0])+'''</td><td>'''+str(lstm_s[x][2])+'''</td>
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
    app.run(debug=False,host="0.0.0.0",port=80)


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
