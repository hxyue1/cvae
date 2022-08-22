import tensorflow as tf
import math
import nsc_tf

def _coupled_div_v1(mean_vector, std_vector, coupling_loss):
      d = mean_vector.shape[1]
      
      d1 = 1 + d*coupling_loss + 2*coupling_loss
      KL_d1 = tf.reduce_prod(
          tf.pow(2 * tf.constant(math.pi, dtype=tf.float64), 
                 coupling_loss/(1 + d*coupling_loss)) \
          * tf.sqrt(d1 / (d1 - 2*coupling_loss*tf.square(std_vector))) \
          * tf.exp(tf.square(mean_vector)*d1*coupling_loss 
                   / (1 + d*coupling_loss) 
                   / (d1 - 2*coupling_loss*tf.square(std_vector))), 
                   1)
      KL_d2 = tf.reduce_prod(
          tf.pow(2 * tf.constant(math.pi, dtype=tf.float64)*tf.square(std_vector),
                 coupling_loss / (1 + coupling_loss*d)) 
          * tf.sqrt(d1 / (1 + d*coupling_loss)), 
          1)

      KL_divergence = (KL_d1 - KL_d2) / coupling_loss / 2
      return KL_divergence

def _kl_div(mean, logvar):
      # Analytical KL-Devergence that I want to put in the code.
      n, m = logvar.shape[0], logvar.shape[1]

      # Create mean and logvar
      mean_ref = tf.constant([0.]*n*m, shape=(n, m), dtype=tf.float64)
      logvar_ref = tf.constant([0.]*n*m, shape=(n, m), dtype=tf.float64)

      # Calculate the determinant of the sample and reference covariance (which are 
      # vectors because the covariance matrices are diagonal).
      det_logvar = tf.reduce_sum(logvar, axis=1)
      det_logvar_ref = 0  # Determinant of the Identity matrix is 1 and log(1) = 0
      # Calculate the log ratio of the determinant of the sample covariance matrix
      # to the determinante of the reference covariance matrix.
      log_det_sigma_div_det_sigma_ref = det_logvar - det_logvar_ref
      # Calculate the trace of the inverse sample covariance matrix times the 
      # reference covariance matrix. For this special case where the new
      # covariance matrix is diagonal and the reference is the identity matrix,
      # This equals the sum of the diagonal of the inverse covariance matrix.
      trace_sigma_inv_sigma_ref = tf.reduce_sum(1 / tf.exp(logvar), axis=1)
      # Calculate the mean difference times the inverse covariance matrix times the
      # mean difference.
      mu_diff_sigma_inv_mu_diff = tf.reduce_sum(
          tf.multiply(tf.pow(mean, 2), tf.math.exp(-1*logvar)),
          axis=1)
      # Calculate the KL-Divergence
      kl_div = 0.5*(log_det_sigma_div_det_sigma_ref - m  + trace_sigma_inv_sigma_ref + mu_diff_sigma_inv_mu_diff)

      return kl_div
  
def neg_elbo_loss(x_recons_logits, x_true, mean, logvar, beta=1.):

      # Sigmoid Cross Entropy Loss
      cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=x_recons_logits,
          labels=x_true
          )

      # Negative Log-Likelihood
      logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3]) # log-likelihood
      neg_ll = -logpx_z  # negative log-likelihood

      # KL-Divergence
      kl_div = _kl_div(mean, logvar)
      
      # ELBO
      neg_ll_mean = tf.math.reduce_mean(neg_ll)
      kl_div_mean = tf.math.reduce_mean(kl_div)
      loss = neg_ll_mean + beta*kl_div_mean

      return_tuple = (loss, neg_ll_mean, kl_div_mean, tf.cast(logpx_z, tf.float64),
                      tf.cast(kl_div, tf.float64))

      return return_tuple

def coupled_neg_elbo(x_recons_logits, x_true, mean, logvar, loss_coupling, beta=1.):
      ##NSC ELBO
      
      #Conversion from logits to probs
      p = x_true
      q = tf.math.sigmoid(x_recons_logits)

      #Calculation of binary log_loss
      cross_ent_2 = p*nsc_tf.math.function.coupled_logarithm(q, 
                                                            kappa=loss_coupling
          ) + (1-p)*nsc_tf.math.function.coupled_logarithm(
                  1-q, 
                  kappa=loss_coupling
                  )

      logpx_z= tf.reduce_sum(cross_ent_2, axis=[1, 2, 3])
      neg_ll = -logpx_z

      kl_div = _coupled_div_v1(mean, tf.exp(logvar/2), loss_coupling)

      neg_ll_mean = tf.math.reduce_mean(neg_ll)
      kl_div_mean = tf.math.reduce_mean(kl_div)

      loss = neg_ll_mean + beta*kl_div_mean

      return_tuple = (loss, neg_ll_mean, kl_div_mean, tf.cast(logpx_z, tf.float64),
                      tf.cast(kl_div, tf.float64))

      return return_tuple

def compute_loss(model, x_true, loss_only=False, loss_coupling=0.0, beta=1.0):
    x_recons_logits, z_sample, mean, logvar = model(x_true)

    if loss_coupling == 0.0:
      loss, neg_ll_mean, kl_div_mean, logpx_z, kl_div = neg_elbo_loss(
          x_recons_logits, x_true, mean, logvar, beta=beta
          )
    else:
      ##NSC ELBO
      loss, neg_ll_mean, kl_div_mean, logpx_z, kl_div = coupled_neg_elbo(
          x_recons_logits, x_true, mean, logvar, loss_coupling, beta=beta
          )

    return_obj = (loss, neg_ll_mean, kl_div_mean, logpx_z, kl_div)

    if loss_only:
        return_obj = loss
    
    return return_obj