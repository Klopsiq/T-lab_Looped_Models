# E005 incident 01 — runtime corruption and resume fix

Документ создан до запуска `runs/E005_v2`.

Первая попытка завершила `relative_fixed16_mlp_first`, seed 23, затем остановилась на шаге 384 auxiliary-траектории. Python runtime сообщил повреждённое имя атрибута `torch.o�es_like`; hashes проекта совпали с локальными, а новый процесс подтвердил наличие рабочего `torch.ones_like`. Инцидент трактуется как transient runtime corruption, а не результат модели.

Повторный запуск обнаружил независимую ошибку пути resume: runner загружал сохранённые CUDA RNG tensors через `map_location=cuda`, тогда как `torch.cuda.set_rng_state_all` в PyTorch 2.10 требует CPU ByteTensor. Исправление добавляет `.cpu()` только при восстановлении RNG и не меняет forward, loss, optimizer, данные или расписание.

Чтобы не менять регистрацию существующего run и не смешивать результаты с разными hashes runner, `runs/E005` сохраняется как прерванная попытка и исключается из итогового анализа. Все четыре траектории запускаются с нуля в `runs/E005_v2`; новая регистрация включает этот документ и исправленный runner.
