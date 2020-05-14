from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from RNN_property_predictor import Model
from feature import molecules
import time
# cannot sure which version of tensorflow is
try:
       import tensorflow.compat.v1 as tf 
       tf.compat.v1.disable_v2_behavior()
except:
       import tensorflow as tf
       

def ECFPNUM_prediction_batch(ls_smi, batch_size=2048, model_IE=None, model_EA=None):
       '''
       this function is to predict IE and EA from ECFPNUM batch by batch, when the amount of data is massive.
       Warning : you will lose some molecules by using this mehtod. 
       In this stage, I separate IE and EA temporily, in the future I consider train a model which predict IE and EA simultaneous
       But, I am not sure whether doing so can lead to a better prediction. 

       Example: 

       model_IE = load_model("model_ECFP/ECFPNUM_IE.h5")
       model_EA = load_model("model_ECFP/ECFPNUM_EA.h5")
       ls_smi = pd.read_csv("OUTPUT")['smiles'].tolist()
       ls_smi_new, IE, EA = ECFPNUM_prediction_batch(ls_smi, batch_size=1024, model_IE=model_IE, model_EA=model_EA)

       '''
       total_num = (len(ls_smi)//batch_size)*batch_size
       epochs = int(total_num/batch_size)
       print('the number of epochs is '+str(epochs))
       start = 0 
       out_IE = []
       out_EA = [] 
       for epoch in range(epochs):
              print(epoch)
              fp_ECFPNUM = molecules(ls_smi[start:start+batch_size]).ECFPNUM()
              out_IE.append(model_IE.predict(fp_ECFPNUM))
              out_EA.append(model_EA.predict(fp_ECFPNUM))
              start += batch_size
       out_IE = np.concatenate(out_IE, axis=0)
       out_EA = np.concatenate(out_EA, axis=0)
    
       return ls_smi[:total_num], out_IE, out_EA


def ECFPNUM_prediction(ls_smi, model_IE=None, model_EA=None):
       '''
       this function is to predict IE and EA from ECFPNUM for all the molecules at once.
       Example: 

       model_IE = load_model("model_ECFP/ECFPNUM_IE.h5")
       model_EA = load_model("model_ECFP/ECFPNUM_EA.h5")
       ls_smi = pd.read_csv("OUTPUT")['smiles'].tolist()
       IE, EA = ECFPNUM_prediction(ls_smi, model_IE=model_IE, model_EA=model_EA)
       '''
       fp_ECFPNUM = molecules(ls_smi).ECFPNUM()
       return model_IE.predict(fp_ECFPNUM), model_EA.predict(fp_ECFPNUM)


def SMILES_onehot_prediction_batch(ls_smi, model_IE=None, model_EA=None, char_set=None, data_MP=None, batch_size=1024):
       '''
       the function is to predict IE and EA from SMILES one-hot encoding batch by batch, when the amount of data is massive. 
       It will return the prediction IE and EA simultaneously. 
       
       Example:
       
       char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
              "s", "O", "[", "Cl", "Br", "\\"]
       data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
       ls_smi = pd.read_csv('OUTPUT.csv')['smiles'].tolist()
       ls_smi_new, IE, EA = SMILES_onehot_prediction_batch(ls_smi, model_name='model_SMILES/model',char_set=char_set, data_MP=data_MP, batch_size=2048)
       '''
       
       #========= normalization =========
       IE = np.asarray(data_MP.values[:,1], dtype=np.float32)
       EA = np.asarray(data_MP.values[:,2], dtype=np.float32) 
       scaler_IE=StandardScaler()
       scaler_IE.fit(IE)

       scaler_EA=StandardScaler()
       scaler_EA.fit(EA)
       
       #=================================
       total_num = (len(ls_smi)//batch_size)*batch_size
       epochs = int(total_num/batch_size)
       print('the number of epochs is '+str(epochs))
       
       tf.reset_default_graph()
       model = Model(seqlen_x=40, dim_x=39, dim_y=2, char_set=char_set)
       
       out_IE = []
       out_EA = []
       ls_smi_new = [] 

       with model.session:
              model.reload(model_name=model_IE)
              start = 0
              for epoch in range(epochs):
                     X, ls_smi_new_batch = molecules(ls_smi[start:start+batch_size]).one_hot(char_set)
                     Y_hat_IE = scaler_IE.inverse_transform(model.predict(X))
                     ls_smi_new.extend(ls_smi_new_batch)
                     out_IE.append(Y_hat_IE)
                     start += batch_size
       
       tf.reset_default_graph()

       with model.session:
              model.reload(model_name=model_EA)
              start = 0
              for epoch in range(epochs):
                     X, ls_smi_new_batch = molecules(ls_smi[start:start+batch_size]).one_hot(char_set)
                     Y_hat_EA = scaler_EA.inverse_transform(model.predict(X))
                     ls_smi_new.extend(ls_smi_new_batch)
                     out_EA.append(Y_hat_EA)
                     start += batch_size
       
       out_IE = np.concatenate(out_IE, axis=0)
       out_EA = np.concatenate(out_EA, axis=0)
       
       return ls_smi_new, out_IE, out_EA


def SMILES_onehot_prediction(ls_smi, model_IE=None, model_EA=None, char_set=None, data_MP=None):
       '''
       the function is to predict IE and EA from SMILES one-hot encoding for all the molecules at once.
       It will return the prediction IE and EA simultaneously. 
       
       Example:
       
       char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
              "s", "O", "[", "Cl", "Br", "\\"]
       data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
       ls_smi = pd.read_csv('OUTPUT.csv')['smiles'].tolist()
       ls_smi_new, IE, EA = SMILES_onehot_prediction(ls_smi, model_name=',char_set=char_set, data_MP=data_MP)
       '''
       
       #========= IE and EA normalization =========
       IE = np.asarray(data_MP.values[:,1], dtype=np.float32).reshape(-1, 1)
       EA = np.asarray(data_MP.values[:,2], dtype=np.float32).reshape(-1, 1)
       scaler_IE=StandardScaler()
       scaler_IE.fit(IE)

       scaler_EA=StandardScaler()
       scaler_EA.fit(EA)
       #===========================================
       tf.reset_default_graph()
       model = Model(seqlen_x=40, dim_x=39, dim_y=1, dim_z=100, dim_h=250, n_hidden=3, batch_size=32, char_set=char_set)

       with model.session:
              model.reload(model_name=model_IE)
              X, ls_smi_new = molecules(ls_smi).one_hot(char_set)
              out_IE = scaler_IE.inverse_transform(model.predict(X))

              model.reload(model_name=model_EA)
              X, ls_smi_new = molecules(ls_smi).one_hot(char_set)
              out_EA = scaler_EA.inverse_transform(model.predict(X))

       
       return ls_smi_new, out_IE, out_EA








        
if __name__ == '__main__':
       
       # start = time.time()
       
       # model_IE = load_model("model_ECFP/ECFPNUM_IE.h5")
       # model_EA = load_model("model_ECFP/ECFPNUM_EA.h5")
       # ls_smi = ['CC(Cl)OCC#N','CC(Cl)CO','CC(C)CCC(F)(F)F','ClCOC1CO1','CCC(C)CC(F)(F)F','OCF','CF']
       # IE, EA = ECFPNUM_prediction(ls_smi, model_IE=model_IE, model_EA=model_EA)

       # data = pd.DataFrame(columns=['smiles', 'IE', 'EA'])
       # data['smiles'] = ls_smi
       # data['IE'] = IE
       # data['EA'] = EA
       # data.to_csv("result_ECFPNUM.csv", index=False)
       
       # end = time.time()
       # print("the execution time "+str(end-start))


       
       # start = time.time()
       
       # model_IE = load_model("model_ECFP/ECFPNUM_IE.h5")
       # model_EA = load_model("model_ECFP/ECFPNUM_EA.h5")
       # ls_smi = pd.read_csv("OUTPUT_multi_latest/all_le_40.csv")['smiles'].tolist()[:100000]
       # ls_smi_new, IE, EA = ECFPNUM_prediction_batch(ls_smi, model_IE=model_IE, model_EA=model_EA)

       # data = pd.DataFrame(columns=['smiles', 'IE', 'EA'])
       # data['smiles'] = ls_smi_new
       # data['IE'] = IE
       # data['EA'] = EA
       # data.to_csv("result_test.csv", index=False)
       
       # end = time.time()
       # print("the execution time "+str(end-start))




       start = time.time()

       char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
              "s", "O", "[", "Cl", "Br", "\\"]
       data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
       ls_smi = data_MP['smiles'].tolist()
       # ls_smi = ['CC(Cl)OCC#N','CC(Cl)CO','CC(C)CCC(F)(F)F','ClCOC1CO1','CCC(C)CC(F)(F)F','OCF','CF']
       
       ls_smi_new, IE, EA = SMILES_onehot_prediction(ls_smi, model_IE='model_RNN/RNN_model_IE.ckpt', 
                                                     model_EA='model_RNN/RNN_model_EA.ckpt',char_set=char_set, data_MP=data_MP)
       # data = pd.DataFrame(columns=['smiles', 'IE', 'EA'])
       # data['smiles'] = ls_smi_new
       # data['IE'] = IE
       # data['EA'] = EA
       # data.to_csv("result.csv", index=False)
       
       # end = time.time()
       # print("the execution time "+str(end-start))


       
       # start = time.time()

       # char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
       #        "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
       #        "s", "O", "[", "Cl", "Br", "\\"]
       # data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
       # ls_smi = pd.read_csv("OUTPUT_multi_latest/all_le_40.csv")['smiles'].tolist()
       # ls_smi_new, IE, EA = SMILES_onehot_prediction_batch(ls_smi, model_name='model_SMILES/model',char_set=char_set, data_MP=data_MP, batch_size=2048)
       
       # data = pd.DataFrame(columns=['smiles', 'IE', 'EA'])
       # data['smiles'] = ls_smi_new
       # data['IE'] = IE
       # data['EA'] = EA
       # data.to_csv("result_substitution.csv", index=False)
       
       # end = time.time()
       # print("the execution time "+str(end-start))






   





