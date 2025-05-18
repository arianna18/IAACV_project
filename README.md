# Sistem cu Inteligență Artificială pentru Detectarea Automată a Minciunii în Vorbire

## Cuprins
1. [Descrierea Proiectului](#descrierea-proiectului)
2. [Tehnologii Utilizate](#tehnologii-utilizate)
3. [Structura Proiectului](#structura-proiectului)
4. [Instalare și Utilizare](#instalare-și-utilizare)
5. [Echipa](#echipa)

---

## Descrierea Proiectului

Acest proiect dezvoltă un sistem avansat de detectare a minciunii prin analiză semnalelor audio, folosind modele de Machine Leaning și rețele neuronale convolutionale (CNN), precum și tehnici moderne de preprocesare.

### Caracteristici Principale:
- Preprocesare a semnalelor audio
- Extragere de trăsături (MFCC, F0 și spectrograme)
- 15 configurații diferite de rețele SVM
- 30 configurații diferite de rețele RF
- 100 configurații diferite de rețele FCNN
- 3 modele a cate 10 configurații diferite de rețele CNN
- Validare încrucișată stratificată (5 fold-uri)
- Optimizare cu early stopping și ajustare automată a ratei de învățare la rețelele CNN

## Tehnologii Utilizate
- Limbaj de programare: Python 3.8+
- Machine Learning: TensorFlow/Keras, Scikit-learn
- Procesare audio: Librosa
- Vizualizare date: Matplotlib
- Management cod: Git

## Structura Proiectului
IAACV_project/
├── extracted_speakers/ # Date brute și preprocesate  
├── processed_files/ # Date procesate pentru a avea lungime egală de 6 secunde
├── spectrograms/ # Spectrograme salvate că .npy  si .png
├── ... # Scripturi python detaliate mai jos
├── requirements.txt # Dependințe
└── README.md # Documentația proiectului

## Instalare și Utilizare

### Cerințe Preliminare
- Python 3.8+
- pip

### Instalare
```bash
gît clone https://github.com/arianna18/IAACV_project
pip install -r requirements.txt
```

### Rularea Proiectului

Pregătirea datelor:
```bash
python3 concatenate_audios.py
python3 create_spectrograms.py
python3 global_normalization.py
```

Antrenarea modelelor:
```bash
python3 svm5.py
python3 rf5.py
python3 cnn_new_new_new.py
```

## Echipa

**Studenți:**
- Diana-Roxana Bratu - diana.bratu@stud.etti.upb.ro
- Iulia-Alexandra Orvas - iulia.orvas@stud.etti.upb.ro
- Arianna Manolache - arianna.manolache@stud.etti.upb.ro

**Profesor coordonator:**
- Șerban Mihalache, PhD - șerban.mihalache@upb.ro