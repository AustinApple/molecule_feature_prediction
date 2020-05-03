# molecule_feature_prediction
These codes are for converting molecule stuctures (SMILES) into different representations(SYBYL, ECFP, ECFP&SYBYL, ECFPNUM, SMILES(one-hot)). And different representation ML models have been trained for the prediction of ionization energy (IE) and electron affinity (EA) by the data from Material project database.
## Tutorial
Convert a list of SMILES into different representation arrays  
```
from feature import molecules

ls_smi = ['C1COC(=O)O1', 'COC(=O)OC', 'O=C(OCC)OCC']
SYBYL = molecules(ls_smi).SYBYL() 
ECFP = molecules(ls_smi).ECFP(radius=2, nbits=2048)
ECFPNUM = molecules(ls_smi).ECFPNUM(radius=2, nbits=2048)

char_set = [" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
              "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
              "s", "O", "[", "Cl", "Br", "\\"]
onehot, ls_smi_new = molecules(ls_smi).one_hot(char_set)
```
Predict IE/EA from different representaions  
```
model_IE = load_model("model_ECFP/ECFPNUM_IE.h5")
model_EA = load_model("model_ECFP/ECFPNUM_EA.h5")
ls_smi = ['C1COC(=O)O1', 'COC(=O)OC', 'O=C(OCC)OCC']
IE, EA = ECFPNUM_prediction(ls_smi, model_IE=model_IE, model_EA=model_EA)
```
If the data is too huge to process, we can use `ECFPNUM_prediction_batch()`  
```
model_IE = load_model("model_ECFP/ECFPNUM_IE.h5")
model_EA = load_model("model_ECFP/ECFPNUM_EA.h5")
ls_smi = pd.read_csv("OUTPUT")['smiles'].tolist()
ls_smi_new, IE, EA = ECFPNUM_prediction_batch(ls_smi, batch_size=1024, model_IE=model_IE, model_EA=model_EA)
```
