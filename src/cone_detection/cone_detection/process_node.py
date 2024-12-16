'''
    Step 1: Get range message for each cone
    Step 2: Add cone locations to position array with inside and outside
        SubStep: Delete all cones outside of range
    Step 3: Find the closest cone on outside to the inside cone
    Step 4: Find the midpoint between the inside and outside cone
    Step 5: Add the midpoint to the way point array
    Step 6: Repeat steps 3-5 until all cones are connected
    Step 7: Return the way point topic
'''