'''
Created on Jun 14, 2011
FP-Growth Algorithm (Frequent Pattern Growth)
Finds frequent itemsets without generating candidate sets (more efficient than Apriori)
Supports Twitter (X) tweet mining for frequent keyword sets
@author: Peter Harrington
Python3 Compatible Version (Fixed: print syntax, twitter library compatibility, robust error handling)
'''
import numpy as np
import re
from time import sleep
# Note: For Twitter API access, install python-twitter (replacement for old twitter library)
# Run: pip install python-twitter
try:
    import twitter
except ImportError:
    print("Warning: 'twitter' library not found. Tweet mining functions will be disabled.")
    print("Install with: pip install python-twitter")

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # Node name (item identifier)
        self.count = numOccur  # Count of transactions passing through this node
        self.nodeLink = None   # Link to next node with the same name (for prefix path search)
        self.parent = parentNode  # Parent node in FP-tree
        self.children = {}     # Child nodes (key: item name, value: treeNode object)
    
    def inc(self, numOccur):
        """Increment node count by numOccur"""
        self.count += numOccur
        
    def disp(self, ind=1):
        """Display FP-tree structure with indentation (for debugging)"""
        # Fixed: Python2 print → Python3 print()
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

def createTree(dataSet, minSup=1):
    """
    Build FP-tree and header table from transaction dataset.
    Args:
        dataSet (dict): Transaction data in format {frozenset(trans): count}
        minSup (int): Minimum support threshold (number of occurrences)
    Returns:
        tuple: (fp_tree, header_table)
            fp_tree: Root node of FP-tree (treeNode object)
            header_table: Header table {item: [support_count, first_node_link]}
    """
    headerTable = {}
    # 1. Count support for each item
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    
    # 2. Prune items below minimum support
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    
    freqItemSet = set(headerTable.keys())
    # Return None if no frequent items exist
    if len(freqItemSet) == 0:
        return None, None
    
    # 3. Enhance header table with node links (initialize to None)
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    
    # 4. Build FP-tree
    retTree = treeNode('Null Set', 1, None)  # Root node (dummy node)
    for tranSet, count in dataSet.items():
        localD = {}
        # Filter and sort items by support (descending)
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # Sort items by support (higher support first)
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # Update FP-tree with sorted items
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    """
    Recursively update FP-tree with sorted items from a transaction.
    Args:
        items (list): Sorted list of frequent items in a transaction
        inTree (treeNode): Current node in FP-tree
        headerTable (dict): Header table with node links
        count (int): Count of the transaction (usually 1 for single transactions)
    """
    # If item exists as child, increment count; else create new node
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        # Create new child node
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # Update header table's node link if first occurrence of this item
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            # Link to existing node with the same name
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    
    # Recursively process remaining items
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    """
    Update node links in header table (link targetNode to the end of nodeToTest's link chain)
    Args:
        nodeToTest (treeNode): Start node of the link chain
        targetNode (treeNode): Node to add to the end of the chain
    """
    while nodeToTest.nodeLink is not None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def ascendTree(leafNode, prefixPath):
    """
    Traverse up from leaf node to root, collecting prefix path (excluding root)
    Args:
        leafNode (treeNode): Leaf node to start traversal
        prefixPath (list): List to store prefix path items
    """
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    """
    Find all conditional pattern bases for a given frequent item (basePat)
    Args:
        basePat (str): Frequent item to find prefix paths for
        treeNode (treeNode): First node link from header table for basePat
    Returns:
        dict: Conditional pattern bases {frozenset(prefix_path): count}
    """
    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        # Prefix path length >1 means there are items before basePat
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        # Move to next node with the same name
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    Recursively mine frequent itemsets from FP-tree and conditional FP-trees
    Args:
        inTree (treeNode): FP-tree to mine
        headerTable (dict): Header table of the FP-tree
        minSup (int): Minimum support threshold
        preFix (set): Prefix set for current frequent items
        freqItemList (list): List to store all frequent itemsets (result)
    """
    # Sort frequent items by support (ascending) to mine in order
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        # Create new frequent itemset by adding basePat to prefix
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        # Get conditional pattern bases for basePat
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # Build conditional FP-tree from pattern bases
        myCondTree, myHead = createTree(condPattBases, minSup)
        # Recursively mine conditional FP-tree if it exists
        if myHead is not None:
            # Uncomment below to print conditional tree structure (debug)
            # print("Conditional Tree for", newFreqSet)
            # myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
    """Load sample transaction dataset for testing FP-Growth"""
    simpDat = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]
    return simpDat

def createInitSet(dataSet):
    """
    Convert raw transaction dataset to dict format {frozenset(trans): count}
    Args:
        dataSet (list): Raw dataset (list of transactions, each transaction is a list)
    Returns:
        dict: Formatted dataset with transaction counts
    """
    retDict = {}
    for trans in dataSet:
        frozenset_trans = frozenset(trans)
        retDict[frozenset_trans] = retDict.get(frozenset_trans, 0) + 1
    return retDict

# ------------------------------
# Twitter (X) Tweet Mining Functions (Optional)
# ------------------------------
def textParse(bigString):
    """
    Parse raw tweet text: remove URLs, split into tokens, filter short tokens
    Args:
        bigString (str): Raw tweet text
    Returns:
        list: Cleaned tokens (lowercase, length > 2)
    """
    # Remove URLs (http://, https://, www.)
    urlsRemoved = re.sub(r'(http:[/][/]|https:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
    # Split into tokens (non-word characters as delimiters)
    listOfTokens = re.split(r'\W+', urlsRemoved)
    # Filter tokens: length > 2, convert to lowercase
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    """
    Fetch tweets from Twitter (X) API using search keyword
    NOTE: 
    1. Replace empty API keys with your own (https://developer.x.com/)
    2. Twitter API requires authentication and has rate limits (free tier is limited)
    3. python-twitter library (v3.3+) is required
    Args:
        searchStr (str): Keyword to search for tweets
    Returns:
        list: List of tweet objects (from twitter API)
    """
    # Replace with your own API credentials (required for access)
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    
    try:
        # Initialize Twitter API client
        api = twitter.Api(
            consumer_key=CONSUMER_KEY,
            consumer_secret=CONSUMER_SECRET,
            access_token_key=ACCESS_TOKEN_KEY,
            access_token_secret=ACCESS_TOKEN_SECRET,
            tweet_mode='extended'  # Get full tweet text (avoid truncation)
        )
    except NameError:
        print("Error: 'twitter' library not installed. Cannot fetch tweets.")
        return []
    except Exception as e:
        print(f"Error initializing Twitter API: {str(e)}")
        return []
    
    resultsPages = []
    max_pages = 5  # Reduced from 14 to avoid rate limit issues
    for i in range(1, max_pages + 1):
        print(f"Fetching tweet page {i}...")
        try:
            # Fetch 100 tweets per page
            searchResults = api.GetSearch(
                term=searchStr,
                count=100,
                page=i,
                result_type='recent'
            )
            if not searchResults:
                print("No more tweets found.")
                break
            resultsPages.extend(searchResults)
            sleep(6)  # Respect API rate limits (6s delay between pages)
        except Exception as e:
            print(f"Error fetching page {i}: {str(e)}")
            break
    print(f"Successfully fetched {len(resultsPages)} tweets.")
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    """
    Mine frequent keyword sets from parsed tweets using FP-Growth
    Args:
        tweetArr (list): List of tweet objects (from getLotsOfTweets)
        minSup (int): Minimum support threshold (number of tweets containing the keyword set)
    Returns:
        list: Frequent keyword sets (sorted by size)
    """
    parsedList = []
    for tweet in tweetArr:
        try:
            # Extract full text from tweet (handle extended tweets)
            if hasattr(tweet, 'full_text'):
                tweet_text = tweet.full_text
            else:
                tweet_text = tweet.text
            # Parse and clean tweet text
            parsedTokens = textParse(tweet_text)
            if parsedTokens:
                parsedList.append(parsedTokens)
        except Exception as e:
            print(f"Error parsing tweet: {str(e)}")
            continue
    
    # Convert parsed tweets to FP-Growth compatible format
    initSet = createInitSet(parsedList)
    # Build FP-tree and mine frequent itemsets
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    if myHeaderTab is not None:
        mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    # Sort frequent itemsets by size (ascending) for readability
    myFreqList.sort(key=len)
    return myFreqList

# ------------------------------
# Example Usage (Runs automatically when executing the script)
# ------------------------------
if __name__ == "__main__":
    # 1. Test FP-Growth with sample transaction data
    print("="*50)
    print("Test 1: FP-Growth with Sample Transaction Data")
    print("="*50)
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    # Mine frequent itemsets with minimum support = 2
    minSup = 2
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    print("FP-Tree Structure:")
    myFPtree.disp()  # Print FP-tree (debug)
    
    freqItemList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), freqItemList)
    print(f"\nFrequent Itemsets (min support = {minSup}):")
    for itemset in freqItemList:
        print(itemset)
    
    # 2. Test Tweet Mining (Uncomment to use, requires Twitter API credentials)
    """
    print("\n" + "="*50)
    print("Test 2: Frequent Keyword Mining from Tweets")
    print("="*50)
    searchKeyword = "python"  # Replace with your target keyword
    tweets = getLotsOfTweets(searchKeyword)
    if tweets:
        frequentKeywords = mineTweets(tweets, minSup=3)
        print(f"\nFrequent Keyword Sets (min support = 3):")
        for kwSet in frequentKeywords:
            print(kwSet)
    """