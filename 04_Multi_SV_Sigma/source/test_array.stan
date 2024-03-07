functions{
  complex_matrix reconstruct_periodogr(matrix re_matrix, matrix im_matrix) {

    complex_matrix[rows(re_matrix), cols(re_matrix)] period_obs;

    for (i in 1:rows(re_matrix)) {
      for (j in 1:cols(re_matrix)) {
        period_obs[i,j] = to_complex(re_matrix[i,j], im_matrix[i,j]); 
      }
    }

    return period_obs;
  }
}

data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> Q;
  // matrix[N, J] data_matrix[Q];
  //array[Q] complex_matrix[N, J] data_matrix;
  array[Q] matrix[N, J] re_matrices;
  array[Q] matrix[N, J] im_matrices;
  // int data_matrix2[Q, N, J]; // specify the same data with data_matrix
  
}

parameters {
  real y;
}

model {
  y ~ normal(0, 1);

  array[Q] complex_matrix[N, J] periodogram;
  for (k in 1:Q) {
    periodogram[Q] = reconstruct_periodogr(re_matrices[Q], re_matrices[Q]);
  }

  //print("dm=", data_matrix); 
  //print("dm[1]=", data_matrix[1]);
  //print("dm[1, 1,]=", data_matrix[1, 1, ]);
  
  // print("dm2[3,2,1]=", data_matrix2[3,2,1]);
  // print("dm[2,1,2]=", data_matrix[2,1,2]);
  // print("dm2[2,1,2]=", data_matrix[2,1,2]);
  // print("dm2=", data_matrix2); 
  // print("dv=", data_vector);
  print("");
} 