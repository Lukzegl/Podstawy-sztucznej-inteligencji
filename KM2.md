# Wstępna Analiza Datasetu 
## Opis Datasetu

Dataset składa się z plików w formacie `.lin`, zawierają one dane z rozgrywek brydżowych. Każdy plik reprezentuje pojedyncze rozdanie. Zawarte informacje:
- Rozdanie kart dla czterech graczy 
- Dealer (gracz rozpoczynający rozdanie)
- Informacja czy dana para jest przed czy po partii
- Sekwencja licytacji z wyjaśnieniami

 
Przykładowy plik `.lin` zawiera:
- Nagłówek z nazwami graczy
- Sekcję `md` z rozdaniem kart
- Sekcję `sv` z vulnerability
- Sekcje `mb` i `an` z licytacją i wyjaśnieniami

##  Przerobienie Pliku .lin na CSV

Proces przekształcania jest zaimplementowany w skrypcie `main.py` z klasami `Class.py` oraz funkcję `save_to_csv` z `tocsv.py`. 

```
S,None,"2S,6S,QS,2H,5H,QH,2D,6D,7D,QD,4C,JC,AC","3S,JS,KS,AS,TH,JH,KH,5D,JD,KD,5C,7C,QC","4S,5S,7S,TS,4H,6H,8H,AH,9D,TD,2C,TC,KC","8S,9S,3H,7H,9H,3D,4D,8D,AD,3C,6C,8C,9C","S:p:Minor suit opening...;W:p:Balanced invite...;..."
```

## Wektory do uczenia modelu

 Dane mogą być używne do przewidywnia wyników licytacji, optymalizacji strategii gry lub analizy siły danej ręki. 

### 1. **Rozkład kart**
   - **One-hot encoding dla każdej karty**: 52 cechy binarne (jedna dla każdej karty w talii).
   - **Rozkład na graczy**: Oddzielne wektory dla 4 raczy.
   - **Punkty za figury** (A=4, K=3, Q=2, J=1) .
   - **Liczba kart w każdym kolorze** 

### 2. **Licytacja**
   - Kodowanie numeryczne  np. 1C=1 1D=2 Pass=0
   - **Długość licytacji** -  liczba odzywek
   - **Pozycja gracza** -  Kodowanie rozdającego i kolejność odzywek


### Przykład wektora
- **Rozdanie**: 4 x 52 = 208 
- **Licytacja**: 40 max długość licytacji x kod 
- **Kontekst**: 8  dealer + vulnerability
- **Razem to ok.** 256

Klasyfikacja - np. poprzez przyznawanie punktów

