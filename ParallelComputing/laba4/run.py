# Запустить из папки laba4: python3 run.py
# Для каждого числа процессов выполняем три замера и берём медиану.
import subprocess
import csv
import statistics
from pathlib import Path

processes = [1,2,4,6,8,12,16]
repeats = 3
Path('results').mkdir(exist_ok=True)
Path('results/times.csv').write_text('processes,vertices,time_s,residual,integral\n')
for p in processes:
    for i in range(repeats):
        print(f'Процессов: {p}, замер {i+1}/{repeats}', flush=True)
        subprocess.run(['mpirun','-n',str(p),'python3','laba4.py'],check=True)

with open('results/times.csv') as f:
    data = list(csv.DictReader(f))
rows = []
for p in processes:
    values = [float(row['time_s']) for row in data if int(row['processes'])==p]
    rows.append([p, statistics.median(values), min(values), max(values)])
print('\nПроцессы | Время, с | Ускорение')
with open('results/table.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['processes','median_s','min_s','max_s','speedup'])
    for p,t,low,high in rows:
        speedup = rows[0][1]/t
        writer.writerow([p,t,low,high,speedup])
        print(f'{p:8d} | {t:8.4f} | {speedup:.3f}')

import matplotlib.pyplot as plt
plt.plot(processes,[row[1] for row in rows],'o-')
plt.xlabel('Число процессов')
plt.ylabel('Время, с')
plt.title('Параллельное решение задачи Пуассона')
plt.xticks(processes)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/time.png',dpi=150)
plt.show()
