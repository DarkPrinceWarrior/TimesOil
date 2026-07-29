# Tech Stack

- Python `>=3.13,<3.14`; управление только `uv`; пакет собирается Hatchling из `src/timesoil`.
- Основные зависимости: NumPy, pandas, SciPy, Matplotlib, openpyxl, PyArrow, pywaterflood, MLForecast, LightGBM.
- Extra `tirex`: chronos-forecasting, tirex-2, PyTorch CPU. Для этого контура использовать `uv sync --extra tirex`; обычный `uv sync` может снять extra.
- Модельный контур: Chronos-2, TiRex-2, TiDE, LightGBM; физика — CRM, двухфазная CRM, CRMP с пластовым давлением, Джентил; ансамбли/стекинг поверх них.
- SPDM/ManiMamba — отдельная исследовательская линия на a100: `external/spdm/.venv`, Python 3.12 + mamba-ssm/CUDA; внешний репозиторий и окружение не входят в git TimesOil.
- Локально WSL — источник кода; тяжёлые численные/GPU-прогоны — `/root/projects/TimesOil` на a100.
- a100: 6× A100-SXM4-40GB без NVLink, P2P выключен. Карты разделяются с production других проектов; свободную карту никогда не предполагать, сначала `nvidia-smi`, чужие сервисы не останавливать.