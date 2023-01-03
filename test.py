from user_input import *

cursor.execute('''  
SELECT * FROM products
          ''')

for row in cursor.fetchall():
    print (row)