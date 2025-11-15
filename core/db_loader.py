import psycopg2
from psycopg2.extras import RealDictCursor

def load_portfolios(conn):
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, userId, techStack, career, projects, activitiesAwards
        FROM "Portfolio"
    """)

    rows = cur.fetchall()
    docs = []

    for r in rows:
        text = f"""
        [Portfolio]
        Tech Stack: {r['techStack'] or ''}
        Career: {r['career'] or ''}
        Projects: {r['projects'] or ''}
        Activities and Awards: {r['activitiesAwards'] or ''}
        """

        meta = {
            "type": "portfolio",
            "id": r["id"],
            "userId": r["userId"]
        }

        docs.append((text, meta))

    return docs


def load_posts(conn):
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT id, authorId, category, title, content
        FROM "Post"
    """)

    rows = cur.fetchall()
    docs = []

    for r in rows:
        text = f"""
        [Post]
        Title: {r['title'] or ''}
        Category: {r['category']}
        Content: {r['content'] or ''}
        """

        meta = {
            "type": "post",
            "id": r["id"],
            "authorId": r["authorId"],
            "category": r["category"]
        }

        docs.append((text, meta))

    return docs
