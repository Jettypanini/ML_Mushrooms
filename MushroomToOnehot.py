import pandas as pd
import numpy as np
#Attribute Information:
#
#1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
#2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
#3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
#4. bruises?: bruises=t,no=f
#5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
#6. gill-attachment: attached=a,descending=d,free=f,notched=n
#7. gill-spacing: close=c,crowded=w,distant=d
#8. gill-size: broad=b,narrow=n
#9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
#10. stalk-shape: enlarging=e,tapering=t
#11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
#12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
#13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
#14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
#15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
#16. veil-type: partial=p,universal=u
#17. veil-color: brown=n,orange=o,white=w,yellow=y
#18. ring-number: none=n,one=o,two=t
#19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
#20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
#21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
#22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

filename = 'mushrooms.csv'
df = pd.read_csv(filename)
####    Calculate all elements  ####
#allElements = []
#for i in range(22):
#    items = []
#    for index, row in df.iterrows():
#        items.append(row[i])
#
#    new_list = list(set(items))
#    allElements.append(new_list)
#print(allElements)

def listToString(s):
 
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for i in range(len(s)):
        str1 += str(int(s[i]))
 
    # return string
    return str1

def listToString2(s):
 
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for i in range(len(s)):
        str1 += str(s[i])
 
    # return string
    return str1

#data = char from csv
#type is number from iterator
def encode(data,type):
    #special case (poisonous is binary encoded)
    if type == 0:
        if data == 'p':
            return "1"
        else:
             return "0"
    body = np.zeros(len(allElements[type])) #create array of zeroes equal to length of parameters per feature
    
    index = 0
    for item in allElements[type]:
        if(data == item):
            body[index] = int(1)
        index += 1
    body = listToString(body)
    return body.replace(".","")
    

allElements = [['p', 'e'], ['f', 'c', 'x', 'k', 's', 'b'], ['f', 'g', 'y', 's'], ['c', 'g', 'u', 'r', 'w', 'e', 'n', 'b', 'p', 'y'], ['f', 't'], ['f', 'c', 'm', 'a', 'l', 's', 'n', 'p', 'y'], ['f', 'a'], ['w', 'c'], ['b', 'n'], ['g', 'u', 'r', 'o', 'k', 'w', 'h', 'e', 'n', 'b', 'p', 'y'], ['t', 'e'], ['c', 'r', '?', 'e', 'b'], ['f', 'k', 'y', 's'], ['f', 'k', 'y', 's'], ['c', 'g', 'o', 'w', 'e', 'n', 'b', 'p', 'y'], ['c', 'g', 'o', 'w', 'e', 'n', 'b', 'p', 'y'], ['p'], ['y', 'w', 'o', 'n'], ['t', 'o', 'n'], ['f', 'e', 'l', 'n', 'p'], ['u', 'o', 'r', 'k', 'w', 'h', 'n', 'b', 'y'], ['c', 'a', 'v', 's', 'n', 'y']]
def main():
    result = ""
    for index, row in df.iterrows():

        #print("item "+ str(index) +":")
        newcode = []
        for i in range(22):
            if len(allElements[i]) > 1:
                temp = encode(row[i],i)
                newcode.append(temp)
        listToString(newcode)
        result += listToString2(newcode) + ","
      
    return result

#print(encode("p",0))
#typeSize = [2,6,4,10,2,9,4,3,2,12,2,7,4,4,9,9,2,4,3,8,9,6,7]
my_string = main()



#print(listToString(encode("e",0)))
#print(my_string)
def calcSize():
    sum = 0
    for i in range (0,len(allElements)):
        sum = sum + len(allElements[i])
        print(len(allElements[i]))
    print(sum)
calcSize()

with open('encodedShroomsV2.csv', 'w') as out:
    out.write(my_string)




    