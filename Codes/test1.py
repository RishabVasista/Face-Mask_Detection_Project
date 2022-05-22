n=int(input())
if n%2!=0 :
 print("Weird")
elif n%2==0 and n>=2 & n<=5 :
 print("Not Weird")
elif n%2==0 and n>=6 & n<=20 :   
  print("Weird")
elif n>=20 and n%2==0 :
    print("Not Weird")