import os
import subprocess

html_content = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>AgentDrone — Командный гайд и дорожная карта</title>
<style>
  @page {
    size: A4;
    margin: 12mm 15mm 12mm 15mm;
  }
  * {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    background: #ffffff;
    line-height: 1.35;
    font-size: 10pt;
    margin: 0;
    padding: 0;
  }
  .page {
    page-break-after: always;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .page:last-child {
    page-break-after: auto;
  }
  .header-block {
    border-bottom: 2px solid #21a038;
    padding-bottom: 8px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }
  .header-title {
    font-size: 18pt;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    margin: 0;
    text-transform: uppercase;
  }
  .header-subtitle {
    font-size: 10.5pt;
    color: #21a038;
    font-weight: 600;
    margin-top: 3px;
  }
  .badge-team {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 8.5pt;
    font-weight: 700;
    color: #334155;
    text-align: right;
  }
  .hero-meta {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }
  .meta-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #21a038;
    border-radius: 4px;
    padding: 6px 8px;
  }
  .meta-card.red {
    border-left-color: #ef4444;
  }
  .meta-card.blue {
    border-left-color: #2563eb;
  }
  .meta-card.purple {
    border-left-color: #8b5cf6;
  }
  .meta-label {
    font-size: 7.5pt;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 700;
  }
  .meta-value {
    font-size: 9.5pt;
    font-weight: 700;
    color: #0f172a;
    margin-top: 2px;
  }
  h2 {
    font-size: 12pt;
    font-weight: 700;
    color: #0f172a;
    margin: 10px 0 6px 0;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 3px;
  }
  h2 span.num {
    background: #21a038;
    color: white;
    font-size: 8pt;
    font-weight: 800;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-right: 6px;
  }
  .pain-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 10px;
  }
  .pain-box {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  .pain-title {
    font-size: 9pt;
    font-weight: 700;
    color: #b91c1c;
    margin-bottom: 4px;
  }
  .pain-desc {
    font-size: 8pt;
    color: #475569;
    line-height: 1.3;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    margin-bottom: 10px;
  }
  th, td {
    padding: 5px 8px;
    border: 1px solid #cbd5e1;
    text-align: left;
  }
  th {
    background: #f1f5f9;
    font-weight: 700;
    color: #1e293b;
  }
  tr:nth-child(even) td {
    background: #f8fafc;
  }
  .badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 7.5pt;
    font-weight: 700;
  }
  .badge-green { background: #dcfce7; color: #15803d; }
  .badge-amber { background: #fef3c7; color: #b45309; }
  .badge-red { background: #fee2e2; color: #b91c1c; }
  .badge-blue { background: #dbeafe; color: #1d4ed8; }
  .badge-purple { background: #f3e8ff; color: #7e22ce; }

  .role-card {
    background: #fff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 8px;
  }
  .role-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px dashed #e2e8f0;
    padding-bottom: 4px;
    margin-bottom: 6px;
  }
  .role-title {
    font-size: 10pt;
    font-weight: 800;
    color: #0f172a;
  }
  .role-focus {
    font-size: 8pt;
    font-weight: 700;
    color: #21a038;
  }
  .role-list {
    margin: 0;
    padding-left: 16px;
    font-size: 8.2pt;
    color: #334155;
  }
  .role-list li {
    margin-bottom: 2px;
  }
  .timeline-row {
    display: flex;
    align-items: stretch;
    margin-bottom: 5px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
  }
  .timeline-time {
    background: #1e293b;
    color: #fff;
    font-weight: 700;
    font-size: 8pt;
    padding: 6px 10px;
    min-width: 110px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
  }
  .timeline-time.highlight {
    background: #21a038;
  }
  .timeline-time.freeze {
    background: #dc2626;
  }
  .timeline-content {
    padding: 6px 10px;
    flex: 1;
  }
  .timeline-title {
    font-size: 8.5pt;
    font-weight: 700;
    color: #0f172a;
  }
  .timeline-sub {
    font-size: 7.8pt;
    color: #64748b;
  }
  .footer {
    border-top: 1px solid #e2e8f0;
    padding-top: 6px;
    font-size: 7.5pt;
    color: #94a3b8;
    display: flex;
    justify-content: space-between;
  }
  .guard-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }
  .guard-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 6px 8px;
    font-size: 8pt;
  }
  .guard-item strong {
    color: #0f172a;
    display: block;
    margin-bottom: 2px;
  }
</style>
</head>
<body>

<!-- СТРАНИЦА 1 -->
<div class="page">
  <div>
    <div class="header-block">
      <div>
        <h1 class="header-title">AgentDrone — Оперативный гайд команды</h1>
        <div class="header-subtitle">AI-агент «Дрон-мониторинг объектов ИЖС» • Команда Memento Vivere</div>
      </div>
      <div class="badge-team">
        ХАКАТОН {SIB}КОД<br>
        31 ОКТ — 1 НОЯ 2026
      </div>
    </div>

    <div class="hero-meta">
      <div class="meta-card">
        <div class="meta-label">Заказчик</div>
        <div class="meta-value">Сибирский банк (Сбер)</div>
      </div>
      <div class="meta-card blue">
        <div class="meta-label">Портфель ИЖС</div>
        <div class="meta-value">17 800 кредитов</div>
      </div>
      <div class="meta-card red">
        <div class="meta-label">Дедлайн кодинга</div>
        <div class="meta-value">Сегодня до 03:00 ночи</div>
      </div>
      <div class="meta-card purple">
        <div class="meta-label">Финал и защита</div>
        <div class="meta-value">1 ноя, 08:00–10:00 (Ведяхину)</div>
      </div>
    </div>

    <h2><span class="num">1</span>Контекст задачи, острая боль Банка и бизнес-эффект</h2>
    <div class="pain-grid">
      <div class="pain-box">
        <div class="pain-title">1. Медленно и дорого (Ногами)</div>
        <div class="pain-desc">
          В Томском пилоте 20 объектов сотрудники на авто осматривали <strong>2 недели</strong> (себестоимость до 30 000 ₽ за отдаленный выезд). Дрон облетает их за <strong>1 день</strong>!
        </div>
      </div>
      <div class="pain-box">
        <div class="pain-title">2. Фрод и слепые зоны</div>
        <div class="pain-desc">
          С земли не видно кровлю и контур. Заемщик может подсунуть фото чужого дома или дешевый щитовой каркас за 800 т.р. вместо согласованного кирпича за 6 млн (нецелевое расходование).
        </div>
      </div>
      <div class="pain-box">
        <div class="pain-title">3. Поздний дефолт (NPL Lag)</div>
        <div class="pain-desc">
          Банк узнает о простое стройки через 2–3 месяца после неплатежа. БПЛА позволяет зафиксировать остановку работ превентивно и спасти залог до банкротства.
        </div>
      </div>
    </div>

    <h2><span class="num">2</span>Нормативные правила Банка и стандарты осмотра</h2>
    <table>
      <thead>
        <tr>
          <th style="width: 25%;">Параметр</th>
          <th style="width: 45%;">Банковский стандарт (из ответов Заказчика и Q&A)</th>
          <th style="width: 30%;">Требование к AI / CV</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Шкала готовности</strong></td>
          <td>
            <strong>0%</strong> — ничего нет (чистое поле/трава);<br>
            <strong>50%</strong> — фундамент, несущие стены, стропила;<br>
            <strong>100%</strong> — закрытый жилой контур (кровля, окна, двери).
          </td>
          <td>Детекция 5 классов конструкций + непрерывный % с маппингом в 0/50/100%.</td>
        </tr>
        <tr>
          <td><strong>Материал стен (Stop-Fraud)</strong></td>
          <td>Кирпич / Камень, Газо/пенобетон, Брус/бревно, Каркасно-щитовой.</td>
          <td>Классификация материала на перспективе 45° для выявления подмены сметы.</td>
        </tr>
        <tr>
          <td><strong>Кадастровая верификация</strong></td>
          <td>
            Имя папки = кадастровый номер (<code>5407055001178</code> &rarr; <code>54:07:055001:178</code>).<br>
            Взлет осуществляется в непосредственной близости к участку.
          </td>
          <td>Сверка GPS взлета с центроидом реестра (&lt;150м) + расчет проекции луча камеры (Pitch/Yaw) на участок.</td>
        </tr>
        <tr>
          <td><strong>Регламент съемки БПЛА</strong></td>
          <td><strong>4 кадра фасадов</strong> (перспектива 40–50°) + <strong>2 сверху</strong> (низкий на кровлю, высокий на границы участка).</td>
          <td>Парсинг <code>Timestamp.MRK</code> и EXIF, валидация полноты ракурсов.</td>
        </tr>
        <tr>
          <td><strong>Выходной артефакт</strong></td>
          <td><strong>«Акт проверки залога» (Приложение 1)</strong> для системы мониторинга залогов.</td>
          <td>Автоматическая генерация PDF с чекбоксами дефектов и фототаблицей.</td>
        </tr>
        <tr>
          <td><strong>Интеграция со Сбером</strong></td>
          <td>Официальный <strong>GigaChat API</strong> (аккаунт физлица по Сбер ID).</td>
          <td>Суммаризация рисков и текстовое заключение инспектора.</td>
        </tr>
      </tbody>
    </table>

    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 8px 12px; margin-top: 6px;">
      <strong style="color: #166534; font-size: 8.5pt;">Ключевое открытие Q&A:</strong>
      <span style="font-size: 8pt; color: #14532d;">
        Заказчик снял требование детального расчета 3-метрового отступа от забора для MVP! Главный критерий победы — <strong>высокая точность CV ядра в оценке готовности (0/50/100%)</strong>, <strong>определение материала стен</strong> и <strong>автоматический выбор одной из 5 стратегий урегулирования</strong>.
      </span>
    </div>
  </div>

  <div class="footer">
    <span>AgentDrone • Memento Vivere</span>
    <span>Страница 1 из 3 — Бизнес-контекст и регламенты</span>
    <span>Хакатон {SIB}КОД 2026</span>
  </div>
</div>

<!-- СТРАНИЦА 2 -->
<div class="page">
  <div>
    <div class="header-block">
      <div>
        <h1 class="header-title">Роли в команде и матрица стратегий</h1>
        <div class="header-subtitle">Декомпозиция ответственности на 3 человек и логика Decision Engine</div>
      </div>
      <div class="badge-team">
        ИТОГО: 3 РАЗРАБОТЧИКА<br>
        ИЗОЛИРОВАННЫЕ ВЕТКИ
      </div>
    </div>

    <h2><span class="num">3</span>Распределение ролей в команде (3 разработчика)</h2>

    <div class="role-card">
      <div class="role-header">
        <span class="role-title">Участник 1: CV & Geo/Data Engineer</span>
        <span class="role-focus">Ветка: feature/cv-pipeline • Файлы: src/ingestion/, src/cv/</span>
      </div>
      <ul class="role-list">
        <li><strong>Парсер телеметрии БПЛА:</strong> чтение EXIF и логов <code>Timestamp.MRK</code> DJI, привязка кадров к 13 кадастровым участкам.</li>
        <li><strong>Детектор полноты съемки:</strong> классификация 6 обязательных ракурсов (4 фасада + 2 верхних кадра).</li>
        <li><strong>CV-ядро готовности:</strong> детекция фундамента, стен, кровли, окон/дверей -> расчет % и маппинг в 0% / 50% / 100%.</li>
        <li><strong>Классификатор материалов стен:</strong> кирпич vs блоки vs дерево vs щитовой каркас на ракурсах 45°.</li>
        <li><strong>Unit-тесты CV:</strong> верификация инференса на тестовых снимках (<code>tests/test_cv.py</code>).</li>
      </ul>
    </div>

    <div class="role-card">
      <div class="role-header">
        <span class="role-title">Участник 2: Backend, Decision Engine & Sber Integrations</span>
        <span class="role-focus">Ветка: feature/decision-engine • Файлы: src/models/, src/engine/, src/report/</span>
      </div>
      <ul class="role-list">
        <li><strong>Pydantic-контракты:</strong> строгие модели <code>CreditObject</code>, <code>DroneImageMeta</code>, <code>InspectionResult</code>.</li>
        <li><strong>Матрица 5 стратегий:</strong> детерминированный движок сопоставления сроков договора, факта CV и просрочки (ПЗ).</li>
        <li><strong>Интеграция GigaChat API:</strong> формирование промпта и получение официального резюме инспектора для системы мониторинга залогов.</li>
        <li><strong>Генератор «Акта проверки залога»:</strong> верстка юридического документа (Приложение 1) в HTML/PDF с фототаблицей.</li>
        <li><strong>Синтетический реестр:</strong> генерация реалистичных данных кредитов для 13 объектов датасета.</li>
      </ul>
    </div>

    <div class="role-card">
      <div class="role-header">
        <span class="role-title">Участник 3: Fullstack/UI & Presentation Lead</span>
        <span class="role-focus">Ветка: feature/streamlit-ui • Файлы: src/ui/, презентация</span>
      </div>
      <ul class="role-list">
        <li><strong>Сквозной Tracer Bullet:</strong> сборка первого сквозного пути «фото -> CV -> стратегия -> Акт -> UI» за первые 4 часа.</li>
        <li><strong>Интерактивный дашборд (Streamlit):</strong> карта 13 объектов (Folium), галерея ракурсов с bounding boxes, статус риска.</li>
        <li><strong>Витрина превентивного мониторинга:</strong> сопоставление проектного графика с фактом до наступления просрочки.</li>
        <li><strong>Кейс Бердска (Супермаркет):</strong> визуализация нецелевого использования земли и препятствий съемки (ЛЭП, деревья).</li>
        <li><strong>Презентация А.А. Ведяхину:</strong> упаковка бизнес-ценности (17,8 тыс. кредитов, ROI, импортонезависимость Сбера).</li>
      </ul>
    </div>

    <h2><span class="num">4</span>Матрица 5 целевых банковских стратегий (Decision Engine)</h2>
    <table>
      <thead>
        <tr>
          <th style="width: 25%;">Стратегия Банка</th>
          <th style="width: 45%;">Условия срабатывания (Входные данные)</th>
          <th style="width: 30%;">Действие системы мониторинга залогов</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><span class="badge badge-green">1. Зеленый коридор</span></td>
          <td>Темп в графике, материалы по проекту, просрочки нет (ПЗ = 0).</td>
          <td>Автосогласование выплаты очередного транша.</td>
        </tr>
        <tr>
          <td><span class="badge badge-amber">2. Реструктуризация</span></td>
          <td>Риск срыва срока (до дедлайна &lt; 6 мес, готовность 50%), задержка транша.</td>
          <td>Инициирование кредитных каникул / дофинансирования.</td>
        </tr>
        <tr>
          <td><span class="badge badge-blue">3. Регистрация залога</span></td>
          <td>Объект готов на 100%, но не зарегистрирован в Росреестре.</td>
          <td>Юридическое содействие заемщику в постановке на залог.</td>
        </tr>
        <tr>
          <td><span class="badge badge-purple">4. Аудит СБ (Фрод)</span></td>
          <td>Подмена материала (щитовой вместо кирпича) или коммерция (магазин).</td>
          <td>Приостановка выплат, назначение выездной проверки СБ.</td>
        </tr>
        <tr>
          <td><span class="badge badge-red">5. Судебное взыскание</span></td>
          <td>Срок договора истекает (&lt; 3 мес), а готовность 0%, либо просрочка 90+ дн.</td>
          <td>Передача дела в суд, реализация участка на торгах.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="footer">
    <span>AgentDrone • Memento Vivere</span>
    <span>Страница 2 из 3 — Роли и Матрица стратегий</span>
    <span>Хакатон {SIB}КОД 2026</span>
  </div>
</div>

<!-- СТРАНИЦА 3 -->
<div class="page">
  <div>
    <div class="header-block">
      <div>
        <h1 class="header-title">Почасовой таймлайн и правила кодинга</h1>
        <div class="header-subtitle">Расписание до сдачи, Executable Guardrails и финальный чек-лист</div>
      </div>
      <div class="badge-team">
        ФИНИШ КОДИНГА: 03:00<br>
        СДАЧА: 10:00 (1 НОЯ)
      </div>
    </div>

    <h2><span class="num">5</span>Почасовой план спринтов (Дедлайны сегодня и завтра)</h2>

    <div class="timeline-row">
      <div class="timeline-time highlight">12:20 — 16:30</div>
      <div class="timeline-content">
        <div class="timeline-title">Спринт 1: Tracer Bullet (Сквозной прототип) — КРИТИЧЕСКИЙ ДЕДЛАЙН!</div>
        <div class="timeline-sub">Pydantic контракты, парсер 1 снимка дрона, эвристика готовности, выбор стратегии, вывод в Streamlit. В 16:30 всё кликается!</div>
      </div>
    </div>

    <div class="timeline-row">
      <div class="timeline-time">16:30 — 19:30</div>
      <div class="timeline-content">
        <div class="timeline-title">Спринт 2: Data Ingestion & Anti-Fraud</div>
        <div class="timeline-sub">Парсер логов <code>Timestamp.MRK</code>, геопривязка к кадастру, проверка 4+2 ракурсов, синтетический реестр 13 объектов НСО.</div>
      </div>
    </div>

    <div class="timeline-row">
      <div class="timeline-time">19:30 — 21:30</div>
      <div class="timeline-content">
        <div class="timeline-title">Спринт 3: Компьютерное зрение (CV Core)</div>
        <div class="timeline-sub">Детекция фундамента/стен/кровли/окон, точный расчет % и нормативные 0/50/100%, классификатор материалов (Stop-Fraud).</div>
      </div>
    </div>

    <div class="timeline-row">
      <div class="timeline-time">21:30 — 23:30</div>
      <div class="timeline-content">
        <div class="timeline-title">Спринт 4: Decision Engine + GigaChat API + Акт осмотра</div>
        <div class="timeline-sub">Полная матрица 5 стратегий, подключение GigaChat (генерация экспертного резюме), автогенерация Акта (Приложение 1 в PDF).</div>
      </div>
    </div>

    <div class="timeline-row">
      <div class="timeline-time">23:30 — 01:30</div>
      <div class="timeline-content">
        <div class="timeline-title">Спринт 5: UI & Вау-фичи для руководства Сбера</div>
        <div class="timeline-sub">Интерактивная карта (Folium), превентивный мониторинг стройки, мультиспектр M3M (сельхоззалоги), кейс супермаркета в Бердске.</div>
      </div>
    </div>

    <div class="timeline-row">
      <div class="timeline-time freeze">01:30 — 03:00</div>
      <div class="timeline-content">
        <div class="timeline-title">Спринт 6: Feature Freeze, Линтеры, Тесты и Запись видео</div>
        <div class="timeline-sub">Прогон <code>pytest</code> и <code>flake8</code> (авто-оценка жюри!), запись резервного скринкаста демо. <strong>В 03:00 — ОТБОЙ НА СОН!</strong></div>
      </div>
    </div>

    <div class="timeline-row">
      <div class="timeline-time highlight">08:00 — 10:00 (1 ноя)</div>
      <div class="timeline-content">
        <div class="timeline-title">Финал: Сборка презентации, репетиция питча и сдача</div>
        <div class="timeline-sub">Слайды под Ведяхина (17,8 тыс. кредитов, 1 день вместо 2 недель, ROI), тайминг речи на 3 мин, загрузка ссылки в форму хакатона.</div>
      </div>
    </div>

    <h2><span class="num">6</span>Паттерны агентной разработки (Agentic Guardrails)</h2>
    <div class="guard-grid">
      <div class="guard-item">
        <strong>1. Tracer Bullet First</strong>
        Сначала пробиваем самый тонкий сквозной путь системы, затем наращиваем точность нейросетей и фичи.
      </div>
      <div class="guard-item">
        <strong>2. One Feature at a Time</strong>
        Берем строго одну задачу из <code>FEATURE_LIST.md</code>, пишем юнит-тест, проверяем, коммитим.
      </div>
      <div class="guard-item">
        <strong>3. Executable Guardrails</strong>
        Перед коммитом: <code>pytest tests/ -v && flake8 src/ tests/</code>. Жюри проверяет качество кода автоматически!
      </div>
      <div class="guard-item">
        <strong>4. Writer — Reviewer</strong>
        Код с AI проверяется вторым участником или свежим агентом на соответствие терминам из <code>DOMAIN.md</code>.
      </div>
    </div>

    <h2><span class="num">7</span>Чек-лист готовности к защите перед жюри</h2>
    <table style="margin-bottom: 0;">
      <tbody>
        <tr>
          <td style="width: 35%;"><strong>1. Работающий прототип</strong></td>
          <td>Выбираем объект из списка -> клик -> через 2 сек видим фото, bounding box, стадию (0/50/100%), стратегию и резюме GigaChat.</td>
        </tr>
        <tr>
          <td><strong>2. Юридический Акт</strong></td>
          <td>Кнопка «Сформировать Акт» отдает готовый красивый PDF по форме Приложения 1 с фототаблицей для системы мониторинга залогов.</td>
        </tr>
        <tr>
          <td><strong>3. Бизнес-подача (Ведяхину)</strong></td>
          <td>Показываем не просто нейросеть, а экономику: <strong>17 800 кредитов ИЖС</strong>, сокращение срока проверки с <strong>14 дней до 1 дня</strong>, ранняя профилактика NPL.</td>
        </tr>
        <tr>
          <td><strong>4. Резервный скринкаст</strong></td>
          <td>Записанное 2-минутное видео работы UI на случай падения интернета / Wi-Fi в зале презентаций.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="footer">
    <span>AgentDrone • Memento Vivere</span>
    <span>Страница 3 из 3 — Таймлайн, Паттерны и Чек-лист</span>
    <span>Хакатон {SIB}КОД 2026</span>
  </div>
</div>

</body>
</html>
"""

html_path = os.path.abspath("TEAM_GUIDE_AND_ROADMAP.html")
pdf_path = os.path.abspath("TEAM_GUIDE_AND_ROADMAP.pdf")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Written HTML to: {html_path}")

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
file_url = "file:///" + html_path.replace("\\", "/")

cmd = [
    chrome_path,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    file_url
]

res = subprocess.run(cmd, capture_output=True, text=True)
print("Chrome return code:", res.returncode)
if os.path.exists(pdf_path):
    print(f"SUCCESS: Generated PDF at {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")
else:
    print("FAILED: PDF not created")
