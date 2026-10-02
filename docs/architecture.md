# Схемы работы локальных моделей

Схемы соответствуют [compose.yaml](../compose.yaml) и
[test_api.py](../test_api.py). Подробное объяснение терминов и настроек —
в [пояснительной записке](explanatory-note.md).

## 1. Общая схема развёртывания

![Общая схема: два независимых API, Docker, модели и постоянные тома](architecture.svg)

[Открыть схему SVG отдельно](architecture.svg). Изображение масштабируется
без потери качества и подходит для вставки в презентацию.

Синим обозначена генерация текста, зелёным — вычисление эмбеддингов.
Стрелки HTTP показывают запрос и ответ. Пунктир обозначает получение
файлов при подготовке; этот путь не участвует в вычислении каждого ответа.

Точная схема сетевых и файловых связей:

```mermaid
flowchart TB
    subgraph internet["Внешние источники: подготовка окружения"]
        Registry["ghcr.io<br/>Образы llama.cpp и TEI"]
        Hub["Hugging Face<br/>Веса, токенизаторы, конфигурации"]
    end

    subgraph mac["Mac: ARM64, 16 ГБ RAM"]
        Client["Клиент в macOS<br/>test_api.py / curl / браузер"]
        Report["artifacts/test-report.json<br/>Результаты проверки"]
        PortLLM["127.0.0.1:8000"]
        PortTEI["127.0.0.1:8080"]

        subgraph vm["Docker Desktop: Linux VM, вычисления на CPU"]
            Engine["Docker Engine + Compose<br/>Образы, контейнеры, сеть, тома"]
            subgraph network["Сеть local-ai_default"]
                LLM["Сервис llm:8080<br/>llama-server + Qwen2.5-7B-Instruct<br/>Текст → текст"]
                TEI["Сервис tei:80<br/>TEI + BGE-M3<br/>Текст → вектор из 1024 чисел"]
            end
            QCache[("local-ai_llm-data<br/>/models: GGUF-файлы")]
            BCache[("local-ai_tei-data<br/>/data: ONNX-файлы и токенизатор")]
        end

        Client <-->|"HTTP / JSON"| PortLLM
        Client <-->|"HTTP / JSON"| PortTEI
        PortLLM <-->|"Проброс 8000 → 8080"| LLM
        PortTEI <-->|"Проброс 8080 → 80"| TEI
        Client -->|"Тест записывает отчёт"| Report
        Engine -.->|"Запускает"| LLM
        Engine -.->|"Запускает"| TEI
        LLM <-->|"Кэш загрузчика и чтение весов"| QCache
        TEI <-->|"Кэш загрузчика и чтение весов"| BCache
    end

    Registry -.->|"Загрузка образов"| Engine
    Hub -.->|"Получает загрузчик llama.cpp"| QCache
    Hub -.->|"Получает загрузчик TEI"| BCache

    classDef llm fill:#eaf2ff,stroke:#2563eb,color:#153365
    classDef tei fill:#e6f6ef,stroke:#16805c,color:#10513c
    class LLM,PortLLM,QCache llm
    class TEI,PortTEI,BCache tei
```

Между `llm` и `tei` в текущем приложении нет потока данных.
Docker создаёт доступную обоим сеть, но не связывает модели автоматически.
Python-тест запускается на Mac и последовательно проверяет оба сервера.

## 2. Первый запуск и повторный запуск

```mermaid
flowchart TD
    Start["docker compose up -d"] --> Config["Прочитать compose.yaml<br/>Подставить .env и переменные окружения"]
    Config --> Images{"Нужные образы уже есть?"}
    Images -->|"Нет"| Pull["Получить образы из ghcr.io"]
    Images -->|"Да"| Create["Создать или обновить контейнеры<br/>Подключить сеть и именованные тома"]
    Pull --> Create
    Create --> QCache{"Qwen есть в /models?"}
    Create --> BCache{"BGE-M3 есть в /data?"}
    QCache -->|"Нет"| QDownload["Скачать обе части GGUF"]
    QCache -->|"Да"| QLoad["Прочитать веса Qwen<br/>Подготовить контекст и KV-кэш"]
    QDownload --> QLoad
    BCache -->|"Нет"| BDownload["Скачать конфигурацию,<br/>токенизатор и ONNX-веса"]
    BCache -->|"Да"| BLoad["Инициализировать backend BGE-M3<br/>Прогреть модель"]
    BDownload --> BLoad
    QLoad --> QReady["llm: /health отвечает успешно"]
    BLoad --> BReady["tei: /health отвечает успешно"]
    QReady --> Test["test_api.py проверяет реальные ответы"]
    BReady --> Test
```

Две ветви подготовки выполняются независимо. `up -d` возвращает управление
после запуска контейнеров и сам по себе не ждёт окончания скачивания весов.
Healthcheck Docker и ожидание в Python-тесте — два отдельных механизма.

При повторном запуске сохранённые веса читаются из тех же томов; RAM и
рабочие кэши процесса подготавливаются заново. Возможные сетевые проверки
метаданных на этой упрощённой схеме не показаны.

## 3. Запрос к Qwen: от вопроса до текста

```mermaid
sequenceDiagram
    autonumber
    participant C as Клиент в macOS
    participant D as Docker: порт 8000 → 8080
    participant S as llama-server
    participant M as Qwen на CPU

    C->>D: POST /v1/chat/completions + JSON
    D->>S: model, messages, temperature, max_tokens
    S->>S: Проверить запрос и применить шаблон диалога
    S->>S: Токенизация входных сообщений
    S->>M: Входные токены: prefill
    M-->>S: Представления входа и состояние KV-кэша
    loop До условия остановки или лимита
        S->>M: Вычислить следующий шаг decode
        M-->>S: Оценки возможных следующих токенов
        S->>S: Выбрать токен и дополнить ответ
    end
    S->>S: Преобразовать токены в текст и собрать JSON
    S-->>D: choices, finish_reason, usage
    D-->>C: HTTP 200 + итоговый JSON
```

В нашем тесте используются `temperature=0`, `max_tokens=16`, `stream=false`.
Получен текст `42`, `finish_reason=stop`, 51 входной и 3 выходных токена.
Модель уже загружена в сервер: клиент не передаёт веса и не загружает их
заново при каждом HTTP-запросе.

## 4. Запрос к BGE-M3: от текста до вектора

```mermaid
flowchart LR
    Input["POST /embed<br/>inputs: четыре текста<br/>normalize: true"]
    Check["TEI<br/>Проверка количества,<br/>токенизация, проверка длины"]
    Batch["Планирование батчей<br/>До 512 токенов<br/>в вычислительном батче"]
    Encoder["BGE-M3<br/>ONNX backend на CPU<br/>Представления токенов"]
    Pool["CLS pooling<br/>Одно представление<br/>каждого текста"]
    Norm["L2-нормализация<br/>Длина вектора ≈ 1"]
    Output["JSON<br/>Четыре вектора<br/>по 1024 числа"]
    Compare["Python<br/>Cosine similarity<br/>и проверка размерности"]

    Input --> Check --> Batch --> Encoder --> Pool --> Norm --> Output --> Compare
```

`/v1/embeddings` предоставляет ту же задачу через другой формат JSON.
Ответ содержит `data`, а у каждого элемента — `index` и `embedding`.
Тест проверяет согласованность этого endpoint с `/embed`.

Смысл контрольного сравнения:

```mermaid
flowchart LR
    RU["Кошка спит на диване."]
    EN["A cat is sleeping on the sofa."]
    OTHER["Вулкан извергает лаву и пепел."]
    SAME["Кошка спит на диване.<br/>Повторный вход"]
    Base["Вектор исходной русской фразы"]
    Translation["Вектор перевода"]
    Different["Вектор посторонней фразы"]
    Duplicate["Вектор повторного входа"]
    RU -->|"BGE-M3"| Base
    EN -->|"BGE-M3"| Translation
    OTHER -->|"BGE-M3"| Different
    SAME -->|"BGE-M3"| Duplicate
    Base ---|"cosine = 0,782093"| Translation
    Base ---|"cosine = 0,374135"| Different
    Base ---|"cosine = 1,000000"| Duplicate
```

Сходство сравнивается между векторами, а не между строками по совпадению букв.
Числа здесь — измерения из [отчёта запуска](../artifacts/test-report.json).
Косинусное сходство не является вероятностью правильности.

## 5. Как Python-тест определяет успех

```mermaid
flowchart TD
    Start["python3 test_api.py"] --> Args["Прочитать параметры CLI<br/>и LLM_URL / TEI_URL"]
    Args --> QWait["Ждать /health Qwen<br/>До --wait секунд"]
    QWait --> QTest["Проверить /v1/models<br/>Вызвать /v1/chat/completions<br/>Проверить формат и ответ 42"]
    QTest --> QResult["Сохранить результат группы Qwen"]
    QWait -.->|"Таймаут / ошибка"| QResult
    QResult --> BWait["Ждать /health TEI<br/>До --wait секунд"]
    BWait --> BTest["Проверить /info и /embed<br/>Размерность, норму, сходство<br/>Проверить /v1/embeddings"]
    BTest --> BResult["Сохранить результат группы BGE-M3"]
    BWait -.->|"Таймаут / ошибка"| BResult
    BResult --> Save["Записать artifacts/test-report.json"]
    Save --> Decision{"Обе группы прошли?"}
    Decision -->|"Да"| Success["passed: true<br/>Код завершения 0"]
    Decision -->|"Нет"| Failure["passed: false<br/>Код завершения 1"]
```

Ожидаемые ошибки во время самих проверок также записываются как результат
со статусом `failed`. Даже если Qwen недоступен, скрипт проверит TEI.
Проверки выполнены последовательно, хотя серверы работают одновременно.

## 6. Возможное расширение: поиск по документам

**Это схема дальнейшего развития. Следующие компоненты не реализованы
в текущем репозитории:** разбиение документов, индекс, поиск и приложение RAG.
Существующие API Qwen и BGE-M3 можно использовать повторно.

```mermaid
flowchart TB
    subgraph indexing["Подготовка документов"]
        Docs["Документы"] --> Chunks["Приложение: разбить на фрагменты"]
        Chunks --> EmbedDocs["Существующий TEI / BGE-M3<br/>Эмбеддинги фрагментов"]
        EmbedDocs --> Index[("Добавить индекс:<br/>векторы + тексты + источники")]
        Chunks -->|"Сохранить исходные тексты"| Index
    end

    subgraph answering["Обработка вопроса"]
        Question["Вопрос пользователя"] --> EmbedQ["Существующий TEI / BGE-M3<br/>Эмбеддинг вопроса"]
        EmbedQ --> Search["Добавить поиск ближайших векторов"]
        Index --> Search
        Search --> Context["Извлечь тексты найденных фрагментов"]
        Context --> Prompt["Приложение собирает:<br/>инструкцию + контекст + вопрос"]
        Question --> Prompt
        Prompt --> LLM["Существующий llama.cpp / Qwen<br/>Генерация по найденному контексту"]
        LLM --> Answer["Ответ пользователю"]
    end
```

Векторы помогают выбрать подходящие фрагменты. В `messages` модели Qwen
приложение передаёт тексты фрагментов и вопрос. Сама установка двух
контейнеров не создаёт этот процесс автоматически.

## Связь схем с файлами

| Элемент на схемах | Где он определён |
| --- | --- |
| Контейнеры, порты, сеть по умолчанию, тома, healthcheck | [compose.yaml](../compose.yaml) |
| Необязательные переопределения портов и образа TEI | [.env.example](../.env.example) |
| HTTP-запросы, валидация и сохранение результата | [test_api.py](../test_api.py) |
| Фактические ответы и значения сходства | [artifacts/test-report.json](../artifacts/test-report.json) |
| Объяснение механизмов, ограничений и параметров | [explanatory-note.md](explanatory-note.md) |
