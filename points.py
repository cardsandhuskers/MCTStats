import math
from sympy import *

# Event Variables
teamSize = 2
teamCount = 10
N = teamSize * teamCount
TOTAL = 4000
print("Team size: ", teamSize)
print("Team count: ", teamCount)
print("Number of Players: ", N)
print("------------------------")

# Math Variables
x = symbols('x')
y = symbols('y')

# Survival Games
def survivalGames(WP, AAE, M):
    """
    Calculates the points the kill points and survival points that 
    should be assigned based on the amount of players.

    
    :param WP: Points assigned to winning team at the end
    :param AAE: Players of the same team alive at the end
    :param M: Kill points are worth M times more than survival points
    """
    # AAE = 2 # Alive At End
    kills = N - AAE

    # using M*y to represent kill points
    a = integrate(y*x, (x,AAE,N))
    equation = Eq(a+(M*kills*y)+WP, TOTAL)

    solution = solve(equation, y)
    decimal_solution = [float(sol.evalf()) for sol in solution]
    survivalPoints = round(decimal_solution[0] * 2) / 2
    killPoints = survivalPoints * M
    
    print("Survival Games")
    print("\tWin Points:", WP)
    print("\tSurvival Points: ", survivalPoints)
    print("\tKill Points: ", killPoints)

# Bingo
def bingo(multiplier):
    """
    Calculates the initial points awarded and the drop
    off for each time an item is collected.

    Assumes that each item can only be collected up to 
    five times for points.

    The predictions for the number of finishes is hard
    to predict. In previous events, the 14/25 items
    were found by four teams (the max), with only
    3/25 items not being found by any teams within
    the alloted times. I'm assuming that by letting
    each item be collected 5 times instead of 4 times,
    there will continue to be a right-skew in the data,
    that peaks at 5 finishes, rather than a more uniform
    distribution between 5 and 4 finishes.

    :param multiplier: How much bigger the initial points
    are compared to the dropOff. initial = drop * multiplier.
    """

    fiveFinishes = 12
    fourFinishes = 4
    threeFinishes = 2
    twoFinishes = 1
    oneFinish = 3

    a = integrate(y-(y*multiplier*x), (x,0,5))
    b = integrate(y-(y*multiplier*x), (x,0,4))
    c = integrate(y-(y*multiplier*x), (x,0,3))
    d = integrate(y-(y*multiplier*x), (x,0,2))
    e = integrate(y-(y*multiplier*x), (x,0,1))

    equation = Eq(fiveFinishes*a+fourFinishes*b+threeFinishes*c+twoFinishes*d+oneFinish*e, TOTAL)
    solution = solve(equation, y)

    decimal_solution = [float(sol.evalf()) for sol in solution]

    initial = round(decimal_solution[0])
    drop = round(initial * multiplier * 2) / 2

    print("Bingo")
    print("\tInitial: ", initial)
    print("\tDrop: ", drop)
    print("\t---------------")
    print("\tFirst Place: ", initial)
    print("\tSecond Place: ", initial-(drop*2))
    print("\tThird Place: ", initial-(drop*3))
    print("\tFourth Place: ", initial-(drop*4))
    print("\tFifth Place: ", initial-(drop*5))
    print("\t~~~~~~~~~~~~~~~")



    a = integrate(initial-(drop*x), (x,0,5))
    b = integrate(initial-(drop*x), (x,0,4))
    c = integrate(initial-(drop*x), (x,0,3))
    d = integrate(initial-(drop*x), (x,0,2))
    e = integrate(initial-(drop*x), (x,0,1))

    equation = Eq(fiveFinishes*a+fourFinishes*b+threeFinishes*c+twoFinishes*d+oneFinish*e, y)

    solution = solve(equation,y)
    decimal_solution = [float(sol.evalf()) for sol in solution]
    print("\tprojected",decimal_solution[0])



# TGTTOS
def otherSide(drop):
    """
    Uses an aggregate of prior event performance to determine the 
    amount of players that will finish each map in the minigame, 
    which is then used to determine the amount of points that should 
    be initially awarded to the first finisher, based on the drop 
    score.

    Based on past performance, maps are divided by difficulty into 
    easy, medium, and hard, with there being 2 of every difficulty 
    for a total of 6 maps. 95% of players finish easy maps, 75% 
    of players finish medium maps, and 60% of players finish
    hard maps.

    :param drop: the amount the initial score is dropped per finisher
    """
    # number of finishers per difficulty type
    easyFinish = round(0.95 * N)
    mediumFinish = round(0.75 * N)
    hardFinish = round(0.6 * N)

    # points awarded per map type
    a = integrate(y-(drop*x), (x,0,easyFinish))
    b = integrate(y-(drop*x), (x,0,mediumFinish))
    c = integrate(y-(drop*x), (x,0,hardFinish))

    # solving for initial points value
    equation = Eq(2*a+2*b+2*c, TOTAL)
    solution = solve(equation, y)
    decimal_solution = [float(sol.evalf()) for sol in solution]
    initial = round(decimal_solution[0])

    print("Get to the Other Side")
    print("\tDrop off:", drop)
    print("\tStart Value: ",initial)
    
def battleBox(deathPercentPerRound):
    """
    Uses an aggregate of kills and wins per round to predict
    the amount of players that will die within each round of 
    battle box. This is used to determine the points that 
    should be awarded for each kill and for each team win.

    Based on past performance, a little less than 50% of total
    players are killed each round, which suggests the general 
    strategy is to kill the opposing team before claiming the
    center, which other variants such as not killing the
    opposing team or a mix of both teams dying before the 
    center is claimed.

    Additionally, not every match results in a win, those this 
    is very rare (about 5% of games)

    This model makes kill points worth 1/5 of the points a team
    receives when winning, similar to past tournaments.

    :param deathPercentPerRound: amount of players that die 
    per round
    """
    # number of rounds based on round robin tournament style
    rounds = teamCount
    if(N % 2 == 0):
        rounds -= 1

    winsPerRound = rounds / 2
    winsPerRound = math.ceil(winsPerRound)

    equation = Eq((rounds*x*winsPerRound)+(deathPercentPerRound*N*winsPerRound*x*(1/5)),TOTAL)
    solution = solve(equation, x)
    decimal_solution = [float(sol.evalf()) for sol in solution]
    winPoints = round(decimal_solution[0])
    killPoints = round(winPoints / 10 * 2)/2

    print("Battle Box")
    print("\tWin Points: ", winPoints)
    print("\tKill Points: " ,killPoints)

def tntRun(first,second,third,rounds):
    """
    Calculates the survival points that should be awarded to 
    every player after each death. 
    
    Based on past performance, this model assumes that there 
    will be exactly one winner at the end of each game.

    :param first: Points awarded to first place (each game)
    :param second: Points awarded to second place (each game)
    :param third: Points awarded to third place (each game)
    :param rounds: Total number of rounds
    """
    a = integrate(x*y, (x,1,N))
    equation = Eq(rounds*(a+first+second+third), TOTAL)
    solution = solve(equation, y)
    decimal_solution = [float(sol.evalf()) for sol in solution]
    survivalPoints = round(decimal_solution[0] * 2) / 2

    print("TNT Run")
    print("\tRounds: 3")
    print("\tFirst Points: ",first)
    print("\tSecond Points: ", second)
    print("\tThird Points: ", third)
    print("\tSurvival Points:", survivalPoints)

def parkour(drop, levelCount):
    """
    Uses an aggregate of past performance to determine the 
    amount of players that will finish the first half of 
    levels, the next 20% of levels, and the remaining 25% 
    of levels. Uses the projected number of completion to
    determine the initial points that should be awarded
    to first place.

    With the past aggregates, 100% of players finish  the 
    first 55% of the levels. 40% of players finish the next
    20% of levels and 18% of players finish the last 25%
    of levels.

    Note that due to the usage of skips, later levels that
    are easier tend to have higher completion rates than
    harder levels that are before them. Skips do not award
    points.

    :param drop: The amount the initial score is dropped per finisher
    :param levelCount: The total number of parkour levels
    """
    # predicted finishers
    aFinishers = N
    bFinishers = round(N * 0.4)
    cFinishers = round(N * 0.18)

    # amount of levels associated with predicted finishes
    aLevels = round(levelCount * 0.55)
    bLevels = round(levelCount * 0.2)
    cLevels = round(levelCount * 0.25)

    count = aLevels + bLevels + cLevels

    # making sure rounded level count equals tootal levels
    while( count != levelCount):
        if(count > levelCount):
            bLevels -= 1
            count -= 1
        elif(count < levelCount):
            bLevels += 1
            count += 1

    # integrals for each section of finishers
    a = integrate(y-drop*(x-1), (x,0,aFinishers))
    b = integrate(y-drop*(x-1), (x,0,bFinishers))
    c = integrate(y-drop*(x-1), (x,0,cFinishers))

    # solving for initial (y)
    equation = Eq((aLevels * a + bLevels * b + cLevels * c), TOTAL)
    solution = solve(equation, y)
    decimal_solution = [float(sol.evalf()) for sol in solution]

    initial = round(decimal_solution[0] * 2) / 2
    lastPlace = initial - ((N-1) * drop)
    print("Parkour")
    print("\tInitial: ", initial)
    print("\tDrop: ", drop)
    print("\tLast place gets: ", lastPlace)

def dropper(drop):
    """
    Uses an aggregate of past performance to determine 
    the amount of players that will finish each map.
    Uses these projections to determine the initial 
    points awarded to the first winner, based on the
    value of the drop points.

    With the past aggregates, 100% of players finish the
    first 5 levels. 90% of players finish the 6th and 7th 
    levels. 80% of players finish the 8th and 9th level.
    70% of players finish the 10th level.

    Note that this model assumes there will be exactly
    10 dropper maps.

    :param drop: The amount the initial score is dropped per finisher
    """
    aFinishers = N
    bFinishers = round(N * 0.9)
    cFinishers = round(N * 0.8)
    dFinishers = round(N * 0.7)

    a = integrate(y-drop*(x-1), (x,0,aFinishers))
    b = integrate(y-drop*(x-1), (x,0,bFinishers))
    c = integrate(y-drop*(x-1), (x,0,cFinishers))
    d = integrate(y-drop*(x-1), (x,0,dFinishers))

    equation = Eq((a * 5 + b * 2 + c * 2 + d), TOTAL)
    solution = solve(equation, y)
    decimal_solution = [float(sol.evalf()) for sol in solution]

    initial = round(decimal_solution[0] * 2) / 2
    lastPlace = initial - ((N-1) * drop)
    print("Dropper")
    print("\tInitial: ", initial)
    print("\tDrop: ", drop)
    print("\tLast place gets: ", lastPlace)


if __name__ == '__main__':
    otherSide(2)
    survivalGames(200,1,10)
    bingo(0.12)
    battleBox(0.45)
    tntRun(50,40,20,3)
    parkour(0.5,30)
    dropper(1)