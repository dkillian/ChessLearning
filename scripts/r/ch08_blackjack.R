# ch08_blackjack.R
# R code from Chapter 8: Blackjack
# Rodriguez & Mendes (2018), Probability, Decisions and Games: A Gentle Introduction using R
#
# Chapter summary:
#   8.1  Rules and Bets
#   8.2  Basic Strategy in Blackjack
#   8.3  A Gambling System that Works: Card Counting
#   8.4  Exercises

# ---- 8.2 Dealer hand distribution (multiple decks, sampling with replacement) ----
# Simulates the dealer's hand to reproduce Table 8.1:
# Probability of each dealer outcome (17, 18, 19, 20, 21, BJ, Bust)
# assuming the dealer hits on anything below 17, played with many decks.

n          <- 100000
cardvalues <- rep(c(seq(1, 10), rep(10, 3)), each = 4)   # 52-card deck values; J/Q/K = 10
outcome    <- rep(0, n)

for (i in 1:n) {
  hand <- sample(cardvalues, 2, replace = TRUE)   # multiple decks -> sample with replacement
  sw   <- TRUE

  while (sw) {
    isace <- (hand == 1)

    if (sum(isace) > 0) {
      # At least one Ace: use 11 if it doesn't bust, otherwise use 1
      if (sum(hand[!isace]) + sum(isace) + 10 > 21) {
        handvalue <- sum(hand[!isace]) + sum(isace)         # all Aces count as 1
      } else {
        handvalue <- sum(hand[!isace]) + sum(isace) + 10    # one Ace counts as 11
      }
    } else {
      handvalue <- sum(hand)
    }

    if (handvalue >= 17) {
      sw <- FALSE                                  # dealer stays
    } else {
      hand <- c(hand, sample(cardvalues, 1))      # dealer hits
    }
  }

  if (handvalue > 21) {
    outcome[i] <- "Bust"
  } else if (handvalue == 21 & length(hand) == 2) {
    outcome[i] <- "BJ"                            # natural blackjack
  } else {
    outcome[i] <- handvalue
  }
}

# Approximate Table 8.1
round(table(outcome) / n, 3)

# Expected output (Table 8.1):
#    17     18     19     20     21     BJ   Bust
# 0.145  0.140  0.134  0.180  0.073  0.047  0.282


# ---- 8.3 Dealer hand distribution (single deck, Aces/2-6 removed) ----
# Illustrates how much probabilities change when low cards are gone from the deck.
# Cards remaining: 7, 8, 9, and 10-valued (J/Q/K/10) only — 28 cards total.
# Sampling WITHOUT replacement (single deck).

n          <- 100000
cardvalues <- rep(c(seq(7, 10), rep(10, 3)), each = 4)   # 28 cards: 7-9, four 10-valued ranks
outcome    <- rep(0, n)

for (i in 1:n) {
  shuffleddeck <- sample(cardvalues, length(cardvalues), replace = FALSE)
  hand         <- shuffleddeck[1:2]
  currentcard  <- 3
  sw           <- TRUE

  while (sw) {
    isace <- (hand == 1)   # no Aces in this depleted deck, but logic retained for generality

    if (sum(isace) > 0) {
      if (sum(hand[!isace]) + sum(isace) + 10 > 21) {
        handvalue <- sum(hand[!isace]) + sum(isace)
      } else {
        handvalue <- sum(hand[!isace]) + sum(isace) + 10
      }
    } else {
      handvalue <- sum(hand)
    }

    if (handvalue >= 17) {
      sw <- FALSE
    } else {
      hand        <- c(hand, shuffleddeck[currentcard])
      currentcard <- currentcard + 1
    }
  }

  if (handvalue > 21) {
    outcome[i] <- "Bust"
  } else if (handvalue == 21 & length(hand) == 2) {
    outcome[i] <- "BJ"
  } else {
    outcome[i] <- handvalue
  }
}

# Approximate Table 8.4
round(table(outcome) / n, 3)

# Expected output (Table 8.4):
#    17     18     19     20     21   Bust
# 0.212  0.186  0.170  0.317  0.001  0.115
# Note: no BJ possible since all Aces are removed
