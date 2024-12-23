# a = input('User Input : ')
# # shortest and longest palindrome
# # non palindroming longest substring
# # list all the available palindrame and non-palindrome (min string length 2)
# string = a.split(' ')
# print(string)

# shortest_palindrome = ''
# longest_palindrome = ''
# non_palindrome = ''



# # for element in string:
# #   temp = []
# #   if eleme
# #   temp.append(element)

# given paranthesis valid or not
square = 0
dict_brac = 0
curv_paren = 0
string = input("User INput: ")
for element in string:
    # print(element)
    if element == '[':
        square += 1
    if element == ']':
        square -=1
    if element == '{':
        dict_brac += 1
    if element == '}':
        dict_brac -=1
    if element == '(':
        curv_paren += 1
    if element == ')':
        curv_paren -=1
    

    
if square != 0:
    print('[] is not in proper format')
if dict_brac != 0:
    print('{} is not in proper format')
if curv_paren != 0:
    print('() is not in proper format')

