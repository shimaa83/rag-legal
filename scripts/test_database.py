from legal_rag.database import get_connection


def main() -> None:
    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT version();")
        version = cursor.fetchone()

    print("PostgreSQL connection successful.")
    print(version[0])


if __name__ == "__main__":
    main()