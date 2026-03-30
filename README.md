# Podstawy-sztucznej-inteligencji
Projekt na PSI
https://docs.google.com/document/d/18Sspqoi3s3ccAYRmeKD5fzJa9tGpEcPRV_hL0zD4kOM/edit?tab=t.0

Autorzy:
Łukasz Żegliński, Kacper Węgrecki
Temat projektu:
Zaprojektowanie modelu AI, który oceni ręce brydżowe i na ich podstawie będzie wybierał poprawne odzywki licytacyjne w Brydżu.






1. Kamień milowy nr 1:
Określenie Celu i Tematu projektu
Dokładne określenie celu projektu, zebranie wymagań, wstępne znalezienie danych które mogą zostać użyte do szkolenia sztucznej inteligencji.
Oczekiwany wynik: Określone wymagania, znalezione zestawy danych które mogą posłużyć do szkolenia AI
2. Kamień milowy nr 2:
Przygotowanie zbioru danych
Przygotowanie, czyszczenie, wstępna analiza danych do treningu, składanie danych z różnych źródeł.
Oczekiwany wynik: gotowy zestaw danych do treningu AI 
3. Kamień milowy nr 3:
Wybór i implementacja modelu AI
Wybór algorytmu, początkowe prace nad implementacją, testowanie różnych rozwiązań na przygotowanym zestawie danych.
Oczekiwany wynik: Gotowy algorytm do dalszej optymalizacji
4. Kamień milowy nr 4:
Ocena wyników i ewentualne poprawki
Optymalizacja wydajności oraz ogólnych wyników, ewentualna poprawa jakości danych, dostosowywanie hiperparametrów
Oczekiwany wynik: Zoptymalizowany model gotowy do wdrożenia do aplikacji
5. Kamień milowy nr 5:
Wdrożenie modelu i przygotowanie końcowej prezentacji
Wdrożenie modelu do aplikacji, integracja z otoczeniem.
Oczekiwany wynik: Gotowy system z którego może korzystać użytkownik.


Pierwszy Kamień milowy - zdefiniowanie projektu:

Celem projektu jest stworzenie modelu AI, który po wytrenowaniu będzie wstanie wskazać poprawne otwarcie licytacji w brydżu (konkretną odzywkę licytacyjną) na podstawie danych wejściowych w postaci ręki brydżowej (13 kart z talii). Dzięki tak wyszkolonemu modelowi będzie można szybko i łatwo uczyć się poprawnej licytacji w brydżu sportowym.
Zasoby wymagane do wdrożenia AI to zbiór danych treningowych w postaci: ręka brydżowa + poprawna odzywka do zalicytowania / końcowy kontrakt.
Wymagania funkcjonalne:
walidacja wejścia - sprawdzenie czy dana ręka jest poprawna
Wybór odpowiedniej odzywki
Zgodność z systemem licytacyjnym
Dane do uczenia:
Dane powinny być zbalansowane (musi być odpowiednio dużo przykładów różnych rozdań)
Odpowiednia duża liczba rozdań
Analiza danych:
NumPy Python
Keras/Pytorch

Możliwe źródła danych:

BigDeal 
https://jfr.pzbs.pl/bigden.htm
Wbridge5
https://wbridge5.software.informer.com/
Bridge base online API
https://www.bridgebase.com/tools/hvdoc.html#:~:text=Multiple%20parameter%20value%20combinations%20must%20be%20separated,an%20ampersand%20character%20(&).%20For%20example:%20//www.bridgebase.com/tools/handviewer.html?

https://lajollabridge.com/Software/BBO-Helper/#:~:text=The%20BBO%20application%20and%20the%20BBO%20standalone,the%20%E2%98%B0%20menu%20in%20the%20History%20pane.

https://www.bridgebase.com/myhands/index.php?&from_login=0

https://www.bridgebase.com/vugraph_archives/vugraph_archives.php

https://www.bridgebase.com/tourneyhistory/
