#include <math.h>     // Per sqrt, cbrt, floor, round, fmax
#include <stdbool.h>  // Per bool, sebbene non strettamente necessario qui
#include <stdio.h>

// Struttura per contenere le coordinate e un codice di errore
typedef struct {
    double x;
    double y;
    double z;
    int    error_code;  // 0 per successo, 1 per input non valido, 2 per errore di calcolo
} Coordinates;

Coordinates xyz_coordinates_from_index_c(int global_idx, int L) {
    Coordinates result = {0.0, 0.0, 0.0, 0};  // Inizializzazione

    // Validazione input
    if (global_idx < 0) {
        fprintf(stderr, "Errore: global_idx (%d) deve essere un intero non negativo.\n", global_idx);
        result.error_code = 1;
        return result;
    }
    if (L < 0) {
        fprintf(stderr, "Errore: L (%d) deve essere un intero non negativo.\n", L);
        result.error_code = 1;
        return result;
    }

    // Caso limite: L == 0
    if (L == 0) {
        if (global_idx == 0) {
            result.x          = 0.0;
            result.y          = 0.0;
            result.z          = 1.0;
            result.error_code = 0;
            return result;
        } else {
            fprintf(stderr, "Errore: Per L=0, global_idx deve essere 0. Ricevuto: %d\n", global_idx);
            result.error_code = 1;
            return result;
        }
    }

    double delta = 1.0 / (double)L;
    double x_root;

    // 1. Determinare k_val
    if (global_idx == 0) {
        x_root = 0.0;
    } else {
        double f_global_idx = (double)global_idx;
        // Risolvi x^3 + 3x^2 + 2x - 6*global_idx = 0
        // Trasformata in X^3 - X - 6*global_idx = 0, con x_root = X_sol - 1
        double discriminant_cubic_sqrt_val = 9.0 * f_global_idx * f_global_idx - (1.0 / 27.0);
        double term_sqrt_val               = sqrt(fmax(0.0, discriminant_cubic_sqrt_val));

        double cbrt_arg1 = 3.0 * f_global_idx + term_sqrt_val;
        double cbrt_arg2 = 3.0 * f_global_idx - term_sqrt_val;

        double x_transformed_root = cbrt(cbrt_arg1) + cbrt(cbrt_arg2);
        x_root                    = x_transformed_root - 1.0;
    }

    // k_val è il più piccolo intero k >= 1 tale che global_idx < N_cumulative(k)
    // N_cumulative(k) = k(k+1)(k+2)/6
    int k_val = (int)floor(x_root + 1e-9) + 1;  // Aggiungi epsilon per stabilità

    // 2. Calcolare z_coord
    result.z = 1.0 - delta * ((double)k_val - 1.0);

    // 3. Determinare idx_in_k_layer
    long long num_pts_before_k_layer = 0;  // Usare long long per prodotti intermedi
    if (k_val > 1) {
        num_pts_before_k_layer = (long long)(k_val - 1) * k_val * (k_val + 1) / 6;
    }
    int idx_in_k_layer = global_idx - (int)num_pts_before_k_layer;

    // 4. Determinare i_loop_val
    // Risolvi i^2 - (2*k_val + 1)*i + 2*idx_in_k_layer = 0 per i
    double term_2k_plus_1          = (double)(2 * k_val + 1);
    double discriminant_i_sqrt_val = term_2k_plus_1 * term_2k_plus_1 - 8.0 * (double)idx_in_k_layer;

    // Controllo opzionale per discriminante negativo (la versione Python lo aveva commentato)
    // if (discriminant_i_sqrt_val < -1e-9) { // Tolleranza per errori float
    //     fprintf(stderr, "Errore: Discriminante negativo per i_loop_val: %f, idx: %d, k_val: %d, idx_in_k: %d\n",
    //             discriminant_i_sqrt_val, global_idx, k_val, idx_in_k_layer);
    //     result.error_code = 2; // Errore di calcolo
    //     return result;
    // }

    double i_loop_val_float = (term_2k_plus_1 - sqrt(fmax(0.0, discriminant_i_sqrt_val))) / 2.0;
    int    i_loop_val       = (int)floor(i_loop_val_float + 1e-9);  // Aggiungi epsilon

    // 5. Calcolare y_coord
    result.y = (double)i_loop_val * delta;

    // 6. Determinare j_loop_val
    long long sum_elements_before_row_i = 0;  // Usare long long per prodotti intermedi
    if (i_loop_val > 0) {
        sum_elements_before_row_i = (long long)i_loop_val * k_val - (long long)i_loop_val * (i_loop_val - 1) / 2;
    }

    // idx_in_k_layer è int, sum_elements_before_row_i è long long.
    // Il risultato j_loop_val_double può essere double.
    double j_loop_val_double = (double)idx_in_k_layer - (double)sum_elements_before_row_i;
    int    j_loop_val        = (int)round(j_loop_val_double);  // round da math.h

    // 7. Calcolare x_coord
    result.x = (double)j_loop_val * delta;

    // Controllo limiti opzionale (la versione Python lo aveva commentato)
    // result.x = fmax(0.0, fmin(1.0, result.x));
    // result.y = fmax(0.0, fmin(1.0, result.y));
    // result.z = fmax(0.0, fmin(1.0, result.z));

    result.error_code = 0;  // Successo
    return result;
}