import time
from main import executar_modelo

intervalo = 3600 * 12  # 12 horas

while True:
    executar_modelo()
    time.sleep(intervalo)