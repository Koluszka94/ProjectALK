# Patient Explorer

GUI w PyQt6 do pracy z danymi pacjentów (CSV/SQLite): filtrowanie, szybkie statystyki i wykresy oraz eksport do CSV/PDF.

## Szybki start
```bash
pip install PyQt6 pandas numpy matplotlib
python app.py
```
W aplikacji kliknij **Browse…** i wskaż plik CSV lub bazę SQLite.

## Funkcje
- Filtry: wiek, płeć, HR, Systolic/Diastolic, objawy.
- Statystyki: count/mean/median/std/min/q1/q3/max + grupowanie (Gender/AgeGroup/Symptoms).
- Wykresy: histogram, scatter, boxplot; zapis PNG. 
- Eksport: CSV oraz raport PDF.
