# Симуляторы и открытые полигоны

Краткий справочник по инструментам гидродинамического моделирования и открытым наборам данных для подготовки суррогатов и проверки контура управления.

## Симуляторы

| Симулятор | Назначение | Доступ |
|-----------|------------|--------|
| **tNavigator** | Промышленный симулятор; исходные модели Model Y / Model Z подготовлены под него | **Коммерческая лицензия** (RFD / Rock Flow Dynamics) |
| **[OPM Flow](https://opm-project.org/?page_id=19)** | Открытый полностью неявный black-oil симулятор, форматы Eclipse/INCLUDE | Свободно, GPL-3 |
| **[MRST](https://www.sintef.no/projectweb/mrst/)** | MATLAB-фреймворк для исследований и прототипов (МRST, SINTEF) | Свободно для академического использования |

Model Z для хакатона приведён к варианту **Model_Z_final_OPM**, чтобы команды могли считать на OPM Flow без привязки к одному вендору.

## Открытые наборы данных

Используйте для обучения суррогатов, отладки генератора `wells_schedule.inc` и сверки методики ЧДД — **не как замену официальных Model Y / Model Z**.

| Набор | Описание | URL |
|-------|----------|-----|
| **OPM Data** | Эталонные модели SPE, Norne и др. для OPM Flow | [github.com/OPM/opm-data](https://github.com/OPM/opm-data) · обзор: [opm-project.org — первый прогон](https://opm-project.org/?page_id=197) |
| **Equinor Volve** | Реальное месторождение, промысловые данные 2008–2016 | [data.equinor.com/dataset/volve](https://data.equinor.com/dataset/volve) |
| **ORSD** (Oil Reservoir Simulations Dataset) | ~60 тыс. прогонов OPM на модели SPE9 для sequence-to-sequence | [developer.ibm.com/exchanges/data/all/oil-reservoir-simulations/](https://developer.ibm.com/exchanges/data/all/oil-reservoir-simulations/) |
| **TPU Reservoir Models** | 1500 black-oil моделей сектора Западной Сибири (ТПУ) | [hw.tpu.ru/en/dataset/](https://hw.tpu.ru/en/dataset/) — доступ по запросу |

## Важно для трека 2

Организаторы **не прикладывают дополнительные дампы прогонов Model Z** для обучения суррогата. Обучающую выборку нужно **сформировать самостоятельно**, прогоняя гидродинамическую модель (OPM Flow, tNavigator или согласованный симулятор) на сценариях управления, которые генерирует ваша мультиагентная система.
