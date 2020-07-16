# molecule_feature_prediction
These codes are for converting molecule stuctures (SMILES) into different representations(SYBYL, ECFP, ECFP&SYBYL, ECFPNUM, SMILES(one-hot)). And different representation ML models have been trained for the prediction of ionization energy (IE) and electron affinity (EA) by the data from Material project database.
## Tutorial
### Convert a list of SMILES into different representation arrays (SYBYL, ECFP, ECFPNUM, SMILES(one-hot))
```
from molecule_feature_prediction.feature import molecules

#ls_smi = ['C1COC(=O)O1', 'COC(=O)OC', 'O=C(OCC)OCC']
ls_smi = pd.read_csv("MP_clean_canonize_cut.csv")['smiles'].tolist()
SYBYL = molecules(ls_smi).SYBYL() 
ECFP = molecules(ls_smi).ECFP(radius=2, nbits=2048)
ECFPNUM = molecules(ls_smi).ECFPNUM(radius=2, nbits=2048)

char_set = [" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
              "s", "O", "[", "Cl", "Br", "\\"]
onehot, ls_smi_new = molecules(ls_smi).one_hot(char_set)
```
### Predict IE/EA from ECFPNUM_NN  
```
from keras.models import load_model
from molecule_feature_prediction.predict import ECFPNUM_NN_prediction

model_IE = load_model("model_ECFP/ECFPNUM_IE.h5")
model_EA = load_model("model_ECFP/ECFPNUM_EA.h5")
ls_smi = ['C1COC(=O)O1', 'COC(=O)OC', 'O=C(OCC)OCC']
IE, EA = ECFPNUM_NN_prediction(ls_smi, model_IE=model_IE, model_EA=model_EA)
```
If the data is too huge to process, we can use `ECFPNUM_NN_prediction_batch()`  
```
from keras.models import load_model
from molecule_feature_prediction.predict import ECFPNUM_NN_prediction_batch

model_IE = load_model("model_ECFP/ECFPNUM_IE.h5")
model_EA = load_model("model_ECFP/ECFPNUM_EA.h5")
ls_smi = pd.read_csv("OUTPUT")['smiles'].tolist()
IE, EA = ECFPNUM_NN_prediction_batch(ls_smi, batch_size=1024, model_IE=model_IE, model_EA=model_EA)
```
### Predict IE/EA from SMILES_RNN
```
from keras.models import load_model
from molecule_feature_prediction.predict import SMILES_onehot_RNN_prediction

char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
        "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
        "s", "O", "[", "Cl", "Br", "\\"]
        
data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
ls_smi = ['CC(Cl)OCC#N','CC(Cl)CO','CC(C)CCC(F)(F)F','ClCOC1CO1','CCC(C)CC(F)(F)F','OCF','CF']

ls_smi_new, IE, EA = SMILES_onehot_RNN_prediction(ls_smi, 
                                                  model_IE='model_RNN/RNN_model_IE.ckpt', 
                                                  model_EA='model_RNN/RNN_model_EA.ckpt',
                                                  char_set=char_set, 
                                                  data_MP=data_MP)
```
If the data is too huge to process, we can use `ECFPNUM_NN_prediction_batch()`
```
from keras.models import load_model
from molecule_feature_prediction.predict import SMILES_onehot_RNN_prediction_batch

char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
        "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
        "s", "O", "[", "Cl", "Br", "\\"]
        
data_MP = pd.read_csv('MP_clean_canonize_cut.csv')
ls_smi = pd.read_csv("OUTPUT")['smiles'].tolist()

ls_smi_new, IE, EA = SMILES_onehot_RNN_prediction(ls_smi, 
                                                  model_IE='model_RNN/RNN_model_IE.ckpt', 
                                                  model_EA='model_RNN/RNN_model_EA.ckpt',
                                                  char_set=char_set, 
                                                  data_MP=data_MP)
```
