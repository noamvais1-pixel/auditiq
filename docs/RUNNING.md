# הרצה

## ללא מפתח

```bash
make demo && make api    # טרמינל 1
make web                 # טרמינל 2
```

האחזור, הסרת המידע המזהה, שער הכיסוי, אכיפת הציטוט ולוג הביקורת פועלים במלואם.
רק הניסוח מוחלף בציטוט ישיר מהמקורות, והממשק מציג באנר.

## עם מודל

```bash
cp .env.example .env
# להוסיף GOOGLE_API_KEY  (או ANTHROPIC_API_KEY)
make ingest              # חובה אחרי החלפת ספק אמבדינגס
make api
```

`LLM_PROVIDER=gemini|anthropic`. `EMBEDDING_PROVIDER=gemini|voyage|openai|local`.

האינדקס נושא את שם הספק שבנה אותו ויסרב להיטען עם ספק אחר, כדי שלא ייווצרו
וקטורים מעורבבים — הכשל השקט ביותר שיש במערכת RAG.

## מכסות ה-free tier של Gemini

רלוונטי אם מריצים את סוויטת ההערכה, שמבצעת כ-13 קריאות יצירה:

| מכסה | ערך |
|---|---|
| יצירה, לכל מודל ליום | **20** |
| אמבדינגס, לדקה | 100 |

**המכסה היא לכל מודל בנפרד.** אם `gemini-3.5-flash` מוצה, מודל אחר עדיין פנוי:

```bash
GEMINI_MODEL=gemini-3.5-flash-lite make eval
```

שתי הגנות בקוד נבנו בעקבות המכסות האלה, ושתיהן שייכות לכל מערכת שמדברת עם API
מארח:

**backoff על 429** — גם ביצירה (`llm/client.py`) וגם באמבדינגס
(`rag/embed_cache.py`), לפי זמן ההמתנה שהשרת עצמו מציע ולא לפי ניחוש מעריכי.

**קאש דיסק לאמבדינגס** — סריקת ארבעה משקלים על 29 שאלות הטמיעה את אותן 29
מחרוזות ארבע פעמים ופוצצה את מכסת הדקה. הקאש ממופתח על (ספק, מודל, task, טקסט);
כל הארבעה נחוצים, כי אותו משפט כ-query וכ-document הוא בכוונה **לא** אותו וקטור.

**זיהוי ריצה מנוונת** — הסוויטה נכשלת אם תשובה כלשהי נפלה לגיבוי. ריצה שנראית
כאילו מדדה את המודל אך מדדה את הגיבוי גרועה מריצה שנכשלה, כי הגיבוי מייצר תשובה
סבירה עם ציטוטים אמיתיים ולכן קשה להבחין בה.

## בעיות נפוצות

**`404 NOT_FOUND` על שם מודל** — לא כל מודל זמין בכל חשבון. לרשימה:

```bash
cd backend && ./.venv/bin/python -c "from dotenv import load_dotenv; load_dotenv('../.env'); import os; from google import genai; [print(m.name) for m in genai.Client(api_key=os.environ['GOOGLE_API_KEY']).models.list() if 'generateContent' in (m.supported_actions or [])]"
```

**`index was built with embedding provider X`** — הריצו `make ingest`.

**`IsADirectoryError` על לוג הביקורת** — היה באג: `AUDIT_LOG_PATH=` ריק ב-.env
נחשב מוגדר-אך-ריק, ולכן ברירת המחדל לא חלה ו-`Path("")` הצביע על הספרייה
הנוכחית. מטופל ב-`config.env()`, שמתייחס לריק כאל לא-מוגדר בכל המשתנים.
