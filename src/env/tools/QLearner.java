package tools;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.logging.*;
import cartago.Artifact;
import cartago.OPERATION;
import cartago.OpFeedbackParam;

public class QLearner extends Artifact {

  private Lab lab; // the lab environment that will be learnt 
  private int stateCount; // the number of possible states in the lab environment
  private int actionCount; // the number of possible actions in the lab environment
  private Map<String,double[][]> qTables; // a map for storing the qTables computed for different goals

  private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

  public void init(String environmentURL) {

    // the URL of the W3C Thing Description of the lab Thing
    this.lab = new Lab(environmentURL);

    this.stateCount = this.lab.getStateCount();
    LOGGER.info("Initialized with a state space of n="+ stateCount);

    this.actionCount = this.lab.getActionCount();
    LOGGER.info("Initialized with an action space of m="+ actionCount);

    qTables = new HashMap<>();
  }

/**
* Computes a Q matrix for the state space and action space of the lab, and against
* a goal description. For example, the goal description can be of the form [z1level, z2Level],
* where z1Level is the desired value of the light level in Zone 1 of the lab,
* and z2Level is the desired value of the light level in Zone 2 of the lab.
* For exercise 11, the possible goal descriptions are:
* [0,0], [0,1], [0,2], [0,3], 
* [1,0], [1,1], [1,2], [1,3], 
* [2,0], [2,1], [2,2], [2,3], 
* [3,0], [3,1], [3,2], [3,3].
*
*<p>
* HINT: Use the methods of {@link LearningEnvironment} (implemented in {@link Lab})
* to interact with the learning environment (here, the lab), e.g., to retrieve the
* applicable actions, perform an action at the lab during learning etc.
*</p>
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  episodesObj the number of episodes used for calculating the Q matrix
* @param  alphaObj the learning rate with range [0,1].
* @param  gammaObj the discount factor [0,1]
* @param epsilonObj the exploration probability [0,1]
* @param rewardObj the reward assigned when reaching the goal state
**/
  @OPERATION
  public void calculateQ(Object[] goalDescription,
                        Object episodesObj,
                        Object alphaObj,
                        Object gammaObj,
                        Object epsilonObj,
                        Object rewardObj) {
      // 0) Parse parameters
      int episodes = Integer.parseInt(episodesObj.toString());
      double alpha = Double.parseDouble(alphaObj.toString());
      double gamma = Double.parseDouble(gammaObj.toString());
      double epsilon = Double.parseDouble(epsilonObj.toString());
      int reward  = Integer.parseInt(rewardObj.toString());

      // 1) Build a List<Object> from the raw Object[]
      List<Object> goalList = new ArrayList<>();
      for (Object o : goalDescription) {
          int g = (o instanceof Number)
                  ? ((Number)o).intValue()
                  : Integer.parseInt(o.toString());
          goalList.add(g);
      }

      // 2) Find all "full" states compatible with that sub‐goal
      List<Integer> goalStates = lab.getCompatibleStates(goalList);
      LOGGER.info("Starting Q-Learning for goal: " + Arrays.toString(goalDescription));
      LOGGER.info("Compatible goal‐states: " + goalStates);
      if (goalStates.isEmpty()) {
          LOGGER.warning("No compatible states found for goal " + Arrays.toString(goalDescription));
          return;
      }

      // 3) Initialize Q‐table
      double[][] qTable = initializeQTable();
      Random random = new Random();

      // 4) Standard Q‐learning over episodes
      for (int e = 0; e < episodes; e++) {
        
        // —————— RANDOMIZE START STATE ——————
        // do a random number of random valid actions
        int randomSteps = random.nextInt(10);    // e.g. up to 10 random moves
        for(int i = 0; i < randomSteps; i++){
            int s0 = lab.readCurrentState();
            List<Integer> possible = lab.getApplicableActions(s0);
            if(possible.isEmpty()) break;
            int a0 = possible.get(random.nextInt(possible.size()));
            lab.performAction(a0);
        }

        // Initialize starting state properly
        int currentState = lab.readCurrentState();
        int step = 0, maxSteps = 200;

        while (!goalStates.contains(currentState) && step++ < maxSteps) {
            // Choose action (epsilon-greedy)
            List<Integer> actions = lab.getApplicableActions(currentState);
            if (actions.isEmpty()) break;
            
            int action;
            if (random.nextDouble() < epsilon) {
                action = actions.get(random.nextInt(actions.size()));
            } else {
                double maxQ = Double.NEGATIVE_INFINITY;
                action = actions.get(0);
                for (int a : actions) {
                    if (qTable[currentState][a] > maxQ) {
                        maxQ = qTable[currentState][a];
                        action = a;
                    }
                }
            }

            // Take action and observe reward
            lab.performAction(action);
            int nextState = lab.readCurrentState();
            double r = calculateReward(goalDescription, reward); // Use your reward function

            // Calculate max Q for next state
            double maxQNext = 0.0;
            if (!goalStates.contains(nextState)) {
                List<Integer> nextActions = lab.getApplicableActions(nextState);
                for (int a : nextActions) {
                    maxQNext = Math.max(maxQNext, qTable[nextState][a]);
                }
            }

            // Update Q-value
            qTable[currentState][action] += alpha * (r + gamma * maxQNext - qTable[currentState][action]);
            
            currentState = nextState;
        }

        if ((e + 1) % 10 == 0) {
            LOGGER.info("Episode " + (e+1) + "/" + episodes + " done");
        }
    }

      // 5) Store under a string key, e.g. "[2, 3]"
      String goalKey = Arrays.toString(goalDescription);
      qTables.put(goalKey, qTable);
      LOGGER.info("Stored Q-table for goal " + goalKey);
      printQTable(goalKey, qTable);
  }

/**
* Returns information about the next best action based on a provided state and the QTable for
* a goal description. The returned information can be used by agents to invoke an action 
* using a ThingArtifact.
*
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  currentStateDescription the current state e.g. [2,2,true,false,true,true,2]
* @param  nextBestActionTag the (returned) semantic annotation of the next best action, e.g. "http://example.org/was#SetZ1Light"
* @param  nextBestActionPayloadTags the (returned) semantic annotations of the payload of the next best action, e.g. [Z1Light]
* @param nextBestActionPayload the (returned) payload of the next best action, e.g. true
**/
  @OPERATION
  public void getActionFromState(Object[] goalDescription,
                                Object[] tags,
                                Object[] values,
                                OpFeedbackParam<String> nextBestActionTag,
                                OpFeedbackParam<Object[]> nextBestActionPayloadTags,
                                OpFeedbackParam<Object[]> nextBestActionPayload) {

    // Build a tag, value map
    Map<String,Object> tv = new HashMap<>();
    for(int i=0;i<tags.length;i++){
      tv.put(tags[i].toString(), values[i]);
    }
    

    //Extract & discretize exactly the seven state dims your Lab expects
    double rawZ1 = ((Number) tv.get("http://example.org/was#Z1Level")).doubleValue();
    double rawZ2 = ((Number) tv.get("http://example.org/was#Z2Level")).doubleValue();
    Boolean  l1    = (Boolean)    tv.get("http://example.org/was#Z1Light");
    Boolean  l2    = (Boolean)    tv.get("http://example.org/was#Z2Light");
    Boolean  b1    = (Boolean)    tv.get("http://example.org/was#Z1Blinds");
    Boolean  b2    = (Boolean)    tv.get("http://example.org/was#Z2Blinds");
    double rawSun = ((Number) tv.get("http://example.org/was#Sunshine")).doubleValue();

    int dZ1     = discretizeLightLevel(rawZ1);
    int dZ2     = discretizeLightLevel(rawZ2);
    int dSun    = discretizeSunshine(rawSun);

    List<Object> currDesc = Arrays.asList(dZ1, dZ2, l1, l2, b1, b2, dSun);
    LOGGER.info("  discrete state  = " + currDesc);

    // Look up the Q‐table
    String goalKey = Arrays.toString(goalDescription);
    double[][] qTable = qTables.get(goalKey);
    if (qTable == null) {
      LOGGER.warning("No Q-table for goal " + goalKey);
      return;
    }

    //Find the matching discrete state index
    List<Integer> matches = lab.getCompatibleStates(currDesc);
    if (matches.isEmpty()) {
      LOGGER.warning("No matching discrete state for " + currDesc);
      return;
    }
    int stateIndex = matches.get(0);

    // pick the one with highest Q
    List<Integer> actions = lab.getApplicableActions(stateIndex);
    if (actions.isEmpty()) {
      LOGGER.warning("No applicable actions at state " + stateIndex);
      return;
    }
    int bestAction = actions.get(0);
    double bestQ = qTable[stateIndex][bestAction];
    for (int a : actions) {
      if (qTable[stateIndex][a] > bestQ) {
        bestQ = qTable[stateIndex][a];
        bestAction = a;
      }
    }

    //Return
    Action action = lab.getAction(bestAction);
    nextBestActionTag.set(action.getActionTag());
    nextBestActionPayloadTags.set(action.getPayloadTags());
    nextBestActionPayload.set(action.getPayload());
  }


  /**
   * Write the full Q-table to a .log file.
   *
   * @param goalKey  the goal (e.g. "[2, 3]"), used in the filename
   * @param qTable   the Q matrix, size [stateCount][actionCount]
   */
  void printQTable(String goalKey, double[][] qTable) {
      // sanitize goalKey for use in filename, e.g. "[2, 3]" → "2_3"
      String safeKey = goalKey
          .replaceAll("\\[|\\]", "")   // remove brackets
          .replaceAll("\\s+", "")      // remove whitespace
          .replace(',', '_');          // comma → underscore
      String filename = "qTable_" + safeKey + ".log";

      try (PrintWriter pw = new PrintWriter(new FileWriter(filename))) {
          // optional header: state index + action columns
          pw.print("State");
          for (int a = 0; a < qTable[0].length; a++) {
              pw.print("\tA" + a);
          }
          pw.println();

          // one line per state
          for (int s = 0; s < qTable.length; s++) {
              pw.print(s);
              for (int a = 0; a < qTable[s].length; a++) {
                  pw.print("\t");
                  pw.print(String.format("%.4f", qTable[s][a]));
              }
              pw.println();
          }

          LOGGER.info("Full Q-table written to " + filename);
      } catch (IOException e) {
          LOGGER.severe("Failed to write Q-table file '" + filename + "': " + e.getMessage());
      }
  }

  /**
  * Initialize a Q matrix
  *
  * @return the Q matrix
  */
 private double[][] initializeQTable() {
    double[][] qTable = new double[this.stateCount][this.actionCount];
    for (int i = 0; i < stateCount; i++){
      for(int j = 0; j < actionCount; j++){
        qTable[i][j] = 0.0;
      }
    }
    return qTable;
  }
 

  // Calculate the reward based on Task 1
  private double calculateReward(Object[] goalDescription, int goalReward) {
    int goalZ1Level = Integer.parseInt(goalDescription[0].toString());
    int goalZ2Level = Integer.parseInt(goalDescription[1].toString());
    
    lab.readCurrentState();
    List<Integer> currentState = lab.currentState;
    
    int z1Level = currentState.get(0);
    int z2Level = currentState.get(1);
    boolean z1Light = currentState.get(2) == 1;
    boolean z2Light = currentState.get(3) == 1;
    boolean z1Blinds = currentState.get(4) == 1;
    boolean z2Blinds = currentState.get(5) == 1;
    int sunshine = currentState.get(6);
    
    double reward = 0.0;
    
    // Achieve target illuminance levels
    boolean goalAchieved = (z1Level == goalZ1Level && z2Level == goalZ2Level);
    if (goalAchieved) {
        return goalReward;
    }
    
    // energy costs as specified in the exercise
    if (z1Light) reward -= 50.0; 
    if (z2Light) reward -= 50.0;
    if (z1Blinds) reward -= 1.0; 
    if (z2Blinds) reward -= 1.0;
    
    // 3. penalties for glare
    boolean isZ1OperatedByHuman = true; // Assuming both workstations are operated by humans
    boolean isZ2OperatedByHuman = true; 
    
    // Check if there is Glare
    if (sunshine == 3) {

      if (isZ1OperatedByHuman && z1Blinds) { 
        reward -= 100.0; 
        LOGGER.info("Glare penalty applied for Zone 1: sunshine level 3 with open blinds");
      }
      
      if (isZ2OperatedByHuman && z2Blinds) {
        reward -= 100.0;
        LOGGER.info("Glare penalty applied for Zone 2: sunshine level 3 with open blinds");
      }
    }
    return reward;
  }

    /**
  * Copied from Lab.java
  * Maps lux values to light levels:
  * lux < 50 -> level 0
  * lux in [50,100) -> level 1
  * lux in [100,300) -> level 2
  * lux >= 300 -> level 3
  */
  private int discretizeLightLevel(Double value) {
    if (value < 50) {
      return 0;
    } else if (value < 100) {
      return 1;
    } else if (value < 300) {
      return 2;
    }
    return 3;
  }

  /**
  * Copied from Lab.java
  * Maps lux values to light levels:
  * lux < 50 -> level 0
  * lux in [50,200) -> level 1
  * lux in [200,700) -> level 2
  * lux >= 700 -> level 3
  */
  private int discretizeSunshine(Double value) {
    if (value < 50) {
      return 0;
    } else if (value < 200) {
      return 1;
    } else if (value < 700) {
      return 2;
    }
    return 3;
  }

  // method to check if target is reached with disrectize values
  @OPERATION
  public void targetReached(Object[] goalDescription,
                          Object[] tags,
                          Object[] values,
                          OpFeedbackParam<Boolean> reached) {
      LOGGER.info("Checking if target is reached for goal: " + Arrays.toString(goalDescription));
      
      //Build map
      Map<String,Object> t = new HashMap<>();
      for(int i=0; i<tags.length; i++){
          t.put(tags[i].toString(), values[i]);
      }
      
      // discretize the light levels for check
      double rawZ1 = ((Number) t.get("http://example.org/was#Z1Level")).doubleValue();
      double rawZ2 = ((Number) t.get("http://example.org/was#Z2Level")).doubleValue();
      
      int currentZ1Level = discretizeLightLevel(rawZ1);
      int currentZ2Level = discretizeLightLevel(rawZ2);
      
      //Get goal light levels
      int goalZ1Level = Integer.parseInt(goalDescription[0].toString());
      int goalZ2Level = Integer.parseInt(goalDescription[1].toString());
      
      //Check
      boolean isGoalReached = (currentZ1Level == goalZ1Level) && (currentZ2Level == goalZ2Level);
      
      LOGGER.info("Current Z1 level: " + currentZ1Level + ", Goal Z1 level: " + goalZ1Level);
      LOGGER.info("Current Z2 level: " + currentZ2Level + ", Goal Z2 level: " + goalZ2Level);
      LOGGER.info("Goal reached: " + isGoalReached);
      
      reached.set(isGoalReached);
  }
}
