'''
Created on Mar 24, 2011
Ch 11: Association Rule Mining with Apriori Algorithm
@author: Peter Harrington
Python3 Compatible Version (No Chinese characters, no votesmart dependency for core functions)
'''
import numpy as np
from time import sleep

def loadDataSet():
    """Return sample transaction dataset for testing Apriori algorithm"""
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    """
    Create C1 (1-item candidate set) from dataset.
    Convert map iterator to list for Python3 compatibility.
    Args:
        dataSet (list): List of transactions (each transaction is a list of items)
    Returns:
        list: C1 (frozenset of 1-item candidates, sorted)
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
    """
    Scan dataset D to filter frequent itemsets (support ≥ minSupport) from Ck.
    Replace dict.has_key() with 'in' (Python3 deprecated has_key).
    Args:
        D (list): List of transactions (each transaction is a frozenset)
        Ck (list): List of k-item candidate sets (frozenset)
        minSupport (float): Minimum support threshold (0~1)
    Returns:
        tuple: (Lk, supportData)
            Lk: List of frequent k-item sets
            supportData: Dict {frequent_itemset: support_value}
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    """
    Generate Ck (k-item candidate sets) from Lk-1 (frequent (k-1)-item sets).
    Merge two sets if their first (k-2) items are identical.
    Args:
        Lk (list): List of frequent (k-1)-item sets
        k (int): Number of items in candidate sets (Ck)
    Returns:
        list: Ck (k-item candidate sets)
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    """
    Main Apriori algorithm: Generate all frequent itemsets and their support values.
    Args:
        dataSet (list): List of transactions (each transaction is a list of items)
        minSupport (float): Minimum support threshold (default: 0.5)
    Returns:
        tuple: (L, supportData)
            L: List of frequent itemsets (L[0]=1-item, L[1]=2-item, ...)
            supportData: Dict {frequent_itemset: support_value}
    """
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):
    """
    Generate strong association rules from frequent itemsets (confidence ≥ minConf).
    Args:
        L (list): List of frequent itemsets
        supportData (dict): Support values of frequent itemsets
        minConf (float): Minimum confidence threshold (default: 0.7)
    Returns:
        list: Strong association rules (tuple: (antecedent, consequent, confidence))
    """
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    Calculate confidence for association rules (freqSet - conseq) → conseq.
    Prune rules with confidence < minConf.
    Args:
        freqSet (frozenset): Frequent itemset
        H (list): List of candidate consequents (frozenset)
        supportData (dict): Support values of frequent itemsets
        brl (list): List to store strong rules
        minConf (float): Minimum confidence threshold
    Returns:
        list: Pruned consequents (confidence ≥ minConf)
    """
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(f"{freqSet - conseq} --> {conseq} conf: {conf:.4f}")
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    Recursively generate longer consequents for association rules.
    Args:
        freqSet (frozenset): Frequent itemset
        H (list): List of candidate consequents (frozenset)
        supportData (dict): Support values of frequent itemsets
        brl (list): List to store strong rules
        minConf (float): Minimum confidence threshold
    """
    m = len(H[0])
    if len(freqSet) > (m + 1):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def pntRules(ruleList, itemMeaning):
    """
    Print association rules in human-readable format.
    Adjust item index to match 1-based item codes in sample data.
    Args:
        ruleList (list): Strong association rules (from generateRules)
        itemMeaning (list): Mapping from (item code - 1) to actual meaning
    """
    for ruleTup in ruleList:
        print("Antecedent: ", end="")
        for item in ruleTup[0]:
            # Adapt 1-based item codes to 0-based list index
            print(itemMeaning[item - 1], end=", ")
        print("\n           -------->")
        print("Consequent: ", end="")
        for item in ruleTup[1]:
            # Adapt 1-based item codes to 0-based list index
            print(itemMeaning[item - 1], end=", ")
        print(f"\nConfidence: {ruleTup[2]:.4f}\n")

# ------------------------------
# Optional: Vote Data Related Functions (No votesmart dependency)
# ------------------------------
def getActionIds():
    """
    [Optional] Fetch vote action IDs from 'recent20bills.txt' (requires votesmart library).
    Returns empty lists if votesmart is not installed.
    """
    print("Warning: 'votesmart' library not found. Cannot fetch real vote data.")
    print("Using simulated vote data for demonstration.")
    return [], []

def getTransList(actionIdList, billTitleList):
    """
    [Optional] Convert real vote data to transaction dataset (requires votesmart library).
    Returns simulated data if votesmart is not installed.
    """
    print("Returning simulated legislative vote transaction data.")
    # Simulated data: [party_code (0=Republican, 1=Democratic), vote_codes (2=Nay, 3=Yea, ...)]
    simulated_trans = [
        [0, 2, 5],   # Republican, Nay on Bill1, Yea on Bill2
        [0, 2, 3],   # Republican, Nay on Bill1, Nay on Bill2
        [1, 3, 5],   # Democratic, Yea on Bill1, Yea on Bill2
        [1, 3, 4],   # Democratic, Yea on Bill1, Nay on Bill2
        [0, 2, 4],   # Republican, Nay on Bill1, Nay on Bill2
        [1, 3, 5],   # Democratic, Yea on Bill1, Yea on Bill2
    ]
    # Item meaning mapping (0-based index for simulated data)
    simulated_item_meaning = [
        'Republican', 'Democratic',
        'Bill1 -- Nay', 'Bill1 -- Yea',
        'Bill2 -- Nay', 'Bill2 -- Yea'
    ]
    return simulated_trans, simulated_item_meaning

# ------------------------------
# Example Usage (Runs automatically when executing the script)
# ------------------------------
if __name__ == "__main__":
    # 1. Test Apriori with sample transaction data
    print("="*50)
    print("Test 1: Apriori with Sample Transaction Data")
    print("="*50)
    sample_data = loadDataSet()
    L_sample, support_sample = apriori(sample_data, minSupport=0.5)
    rules_sample = generateRules(L_sample, support_sample, minConf=0.7)
    # Print rules (itemMeaning matches 1-based item codes in sample data)
    pntRules(rules_sample, itemMeaning=['Item1', 'Item2', 'Item3', 'Item4', 'Item5'])

    # 2. Test Apriori with simulated legislative vote data
    print("\n" + "="*50)
    print("Test 2: Apriori with Simulated Vote Data")
    print("="*50)
    action_ids, bill_titles = getActionIds()
    vote_trans, item_meaning = getTransList(action_ids, bill_titles)
    # Run Apriori (lower minSupport for sparse vote data)
    L_vote, support_vote = apriori(vote_trans, minSupport=0.3)
    rules_vote = generateRules(L_vote, support_vote, minConf=0.8)
    # Print vote-related rules (party → voting behavior)
    pntRules(rules_vote, item_meaning)