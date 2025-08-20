# TFM Máster en Ingeniería Matemática - Lluís Boscà

Este repositorio contiene el código empleado en mi Trabajo de Fin de Máster, titulado **"Valoración de opciones asiáticas"**, del Máster en Ingeniería Matemática de la Universidad Complutense de Madrid. Incluye implementaciones en Python de los distintos métodos numéricos desarrollados en el trabajo, así como scripts para la ejecución y evaluación de los resultados.

## Contenido del Repositorio

La carpeta `src/` contiene los siguientes archivos:

- **Asian_NM_Lib.py**  
  Módulo principal con la implementación de todos los métodos numéricos utilizados:  
  - Diferencias finitas con esquema de Crank–Nicolson (media aritmética).  
  - Simulaciones de Montecarlo con trayectorias antitéticas (media aritmética y geométrica).  
  - Fórmulas exactas cerradas para el caso geométrico (Kemna y Vorst).  

- **Asian_avgPrice_plot.py**  
  Calcula precios de opciones Call asiáticas con **precio promedio**, variando la volatilidad, y genera una **gráfica comparativa** (PDF) entre diferencias finitas, Montecarlo y la fórmula exacta en el caso geométrico.

- **Asian_avgPrice_table.py**  
  Evalúa precios de opciones Call asiáticas con **precio promedio** para distintas combinaciones de strike, volatilidad y tipo de interés. Los resultados se guardan en un **archivo CSV**.

- **Asian_avgStrike_plot.py**  
  Calcula precios de opciones Call asiáticas con **strike promedio**, variando la volatilidad, y genera una **gráfica comparativa** (PDF) entre diferencias finitas, Montecarlo y la fórmula exacta en el caso geométrico.

- **Asian_avgStrike_table.py**  
  Evalúa precios de opciones Call asiáticas con **strike promedio** para distintas combinaciones de volatilidad y tipo de interés. Los resultados se guardan en un **archivo CSV**.

- **Asian_comp_execTimes.py**  
  Mide y compara el **tiempo medio de ejecución** de los distintos métodos (diferencias finitas y Montecarlo) tanto para opciones de strike promedio como de precio promedio. Exporta los resultados en un **CSV**.

- **AsianAvgPriceCall_vs_EuroCall.py**  
  Compara precios de opciones asiáticas de **precio promedio** (medias aritmética y geométrica) frente a una **opción europea vanilla**, variando el precio inicial del subyacente. Genera una **gráfica comparativa** con el payoff incluido.
