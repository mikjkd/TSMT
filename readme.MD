Il progetto seguente è impostato per crare dei predittori che lavorano su serie temporali legate a rilevazioni di Radon.

##
I file `dataset.py` e `data_plot.py` servono a creare il dataset e plottare le sue caratteristiche, e sono utilizzabili 
in maniera autonoma essendo provvisti di main.

##
I file `data_generator.py` e `model.py` contengono:
* la classe Dataset, responsabile di creare i data generator utilizzati per il training
* la classe Model, la quale consente di creare il modello di regressione, con determinate caratteristiche e basato sui data shape di Dataset
* la classe Trainer, che consente l'allenamento del modello tramite i data generator.

Le tre classi vengono utilizzate nel file `main.py`, nel quale è presente la creazione del dataset fisico a partire da un file csv,
la creazione degli oggetti dataset,model e trainer, con allenamento del modello.

## 
Il file `eval_model.py` contiene il codice per valutare il modello allenato con le caratteristiche descritte in `main.py`
e vengono misurate una serie di metriche come la loss sul test set, ed il coefficiente di correlazione di Pearson.

