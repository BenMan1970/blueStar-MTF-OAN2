# ==================== CONFIG OPTIMISÉE (INSTITUTIONNEL) ====================

# On rééquilibre pour donner la priorité à la tendance "Actionnable" (H4/D)
# Le Monthly donne le biais de fond, mais ne doit pas étouffer le signal Daily.
MTF_WEIGHTS = {
    '15m': 1.0, 
    '1H':  2.0, 
    '4H':  4.0,  # Augmenté (Zone de structure)
    'D':   5.0,  # Augmenté (Le Roi du Swing)
    'W':   3.0,  # Contexte
    'M':   2.0   # Biais de fond (réduit pour ne pas lagger)
}
TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())

# On garde les confirmations pour filtrer le bruit
CONFIRMATION_BARS = {'15m': 2, '1H': 2, '4H': 2, 'D': 1, 'W': 1, 'M': 1}


# ==================== BLUSTAR ENGINE V2 (INSTITUTIONNEL) ====================
def calc_professional_trend(df, timeframe='D'):
    if df.empty or len(df) < 200: # On a besoin de plus d'historique pour la 200
        return 'Neutral', 0, 'C', 0

    close = df['Close']
    high = df['High']
    low = df['Low']

    # 1. INDICATEURS TECHNIQUES
    # ZLEMA (Tendance rapide / Entrée)
    zlema_s = zlema(close, LENGTH) 
    
    # BASELINE (Tendance de fond Institutionnelle - EMA 200)
    baseline = close.ewm(span=200, adjust=False).mean()

    # CANAL DE VOLATILITÉ (Pour éviter les faux signaux dans le bruit)
    atr_a = atr(high, low, close, VOLATILITY_PERIOD)
    volatility = atr_a.rolling(LENGTH).mean() * MULT
    upper = zlema_s + volatility
    lower = zlema_s - volatility

    # ADX (Uniquement pour la FORCE, pas la DIRECTION)
    adx_val, plus_di, minus_di = adx(high, low, close)
    last_adx = adx_val.iloc[-1]

    # 2. LOGIQUE DIRECTIONNELLE (C'est ici que ça change tout)
    # On regarde la relation entre Prix, ZLEMA et Baseline
    
    current_price = close.iloc[-1]
    last_zlema = zlema_s.iloc[-1]
    last_baseline = baseline.iloc[-1]
    
    # Détection Trend Primaire
    trend_state = 'Neutral'
    
    # BULLISH SCENARIOS
    if current_price > last_zlema:
        if current_price > last_baseline:
            trend_state = 'Bullish' # Strong Trend
        else:
            # Prix au dessus de ZLEMA mais sous la 200 = Rebond ou Début de retournement
            # Si le momentum est fort (DI+ > DI-), on valide le Bullish
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                trend_state = 'Bullish'
            else:
                trend_state = 'Range' # Probable compression
                
    # BEARISH SCENARIOS
    elif current_price < last_zlema:
        if current_price < last_baseline:
            trend_state = 'Bearish' # Strong Trend
        else:
            # Prix sous ZLEMA mais au dessus de la 200 = Correction
            if minus_di.iloc[-1] > plus_di.iloc[-1]:
                trend_state = 'Bearish'
            else:
                trend_state = 'Range'

    # 3. CONFIRMATION HYSTERESIS (Anti-fakeout)
    # On vérifie si la structure des dernières bougies contredit la logique
    # Si on est Bullish, on ne veut pas de Lower Low sur les 3 dernières bougies
    if trend_state == 'Bullish':
        if close.iloc[-1] < close.iloc[-2] and close.iloc[-2] < close.iloc[-3]:
            # Petit repli technique, on garde Bullish mais on note
            pass 

    # 4. CALCUL DE LA FORCE (QUALITY)
    # La qualité dépend de l'alignement Prix / ZLEMA / Baseline + ADX
    
    score = 0
    # Alignement parfait (Le Setup "Royal")
    if trend_state == 'Bullish' and current_price > last_zlema and last_zlema > last_baseline:
        score += 50
    elif trend_state == 'Bearish' and current_price < last_zlema and last_zlema < last_baseline:
        score += 50
    else:
        score += 20 # Tendance présente mais conflit avec la baseline
        
    # Bonus ADX (Momentum)
    score += min(50, last_adx) 

    # Attribution Note
    if score >= 80: quality = 'A+' # Tendance forte alignée
    elif score >= 60: quality = 'A'
    elif score >= 40: quality = 'B'
    else: quality = 'C' # Tendance faible ou Range

    # Si l'ADX est vraiment mort (< 12), on force Range car pas de liquidité
    if last_adx < 12:
        trend_state = 'Range'
        quality = 'C'

    return trend_state, round(score, 1), quality, round(last_adx, 1)
