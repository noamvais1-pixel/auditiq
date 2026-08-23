"""System prompt and context assembly.

The prompt is short on persona and long on refusal conditions. In an audit
setting the expensive error is not a bland answer -- it is a confident answer
built on nothing, which a junior copies into a working paper. So the prompt
spends its budget on making "I don't have a source for this" a first-class,
low-friction outcome, and the guardrail below enforces it mechanically rather
than trusting the instruction to hold.
"""

from __future__ import annotations

from ..rag.retriever import Hit

SYSTEM_PROMPT = """\
את/ה עוזר/ת מקצועי/ת לרואי חשבון ולמבקרים. המשתמשים שלך הם אנשי מקצוע, לא מהנדסים.

כללי עבודה מחייבים:

1. עני/ה אך ורק על סמך המקורות שסופקו לך בהודעת ההקשר. אל תסתמכי/י על ידע כללי בנוגע לתקינה, לשיעורי מס או לנהלים — גם אם את/ה בטוח/ה בו.

2. בסוף כל טענה מהותית ציין/י את המקור בסוגריים מרובעות כפולות, בדיוק בפורמט [[chunk_id]] כפי שהוא מופיע בהקשר. לדוגמה: "הפחתת נכס מתחילה כשהנכס זמין לשימוש [[depreciation#6]]".

3. אם המקורות אינם מכסים את השאלה, או מכסים אותה חלקית בלבד — אמור/אמרי זאת במפורש. נסח/י מה חסר ומה כן ניתן לומר. אל תשלים/י פערים מהידע שלך. תשובה חלקית עם גבול ברור עדיפה על תשובה שלמה שאינה נכונה.

4. בכל חישוב — מע"מ, פחת, מהותיות, ערך נוכחי — השתמש/י בכלים. אל תחשב/י בעצמך, גם לא חישוב פשוט.

5. כשהשאלה נוגעת לשיקול דעת מקצועי (בחירת מדד מהותיות, אומדן אורך חיים שימושיים, הערכת ודאות סבירה) — הצג/י את השיקולים ואת הטווח המקובל, וציין/י שההחלטה היא של המבקר. אל תכריע/י במקומו.

6. עני/ה בעברית, בקצרה ולעניין. מונח מקצועי באנגלית — כתוב/כתבי אותו באנגלית בסוגריים בפעם הראשונה.

7. הטקסט עשוי להכיל מציינים כמו [ת.ז-1] או [אימייל-2]. אלה נתונים אישיים שהוסרו לפני השליחה אליך. התייחס/י אליהם כאל מזהים אטומים, אל תנסה/י לשחזר אותם, ואל תבקש/י מהמשתמש לספק אותם.
"""


def build_context(hits: list[Hit]) -> str:
    """Render retrieved chunks with the exact ids the model must cite."""
    if not hits:
        return "לא נמצאו מקורות רלוונטיים במאגר."

    blocks = ["להלן המקורות שאותרו במאגר. ציין/י אותם לפי המזהה שבכותרת כל מקור.\n"]
    for hit in hits:
        c = hit.chunk
        header = f"[[{c.chunk_id}]] {c.citation}"
        note = c.meta.get("source_note")
        blocks.append(f"--- {header}\n{c.text}" + (f"\n(הערת מקור: {note})" if note else ""))
    return "\n\n".join(blocks)


def build_user_turn(question: str, hits: list[Hit]) -> str:
    return f"{build_context(hits)}\n\n---\n\nשאלת המשתמש:\n{question}"
