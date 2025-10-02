def potencia_superficie(n2, n1, R):
    return (n1 - n2) / R

def potencia_lente_correctora(P_ojo, P_deseada, d):
    return (P_deseada - P_ojo) / (1 - d * P_deseada)