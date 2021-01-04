import parameters as par
import database_io as db

# Start session and get add data to database from csv files.
session = db.begin_session()

# Create table if it doesnt exist.
try:
    db.create_table(session)
except:
    print("Table already exists.")

# Add content to table.
db.add_content_csv(session)

# Show all the content.
db.show_content(session)
