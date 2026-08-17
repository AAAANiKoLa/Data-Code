%% Code documentation description
% This code simulates a workflow for 
% "compressive optical transmission matrix imaging through dynamic scattering".
% It models the construction of a transmission matrix and 
% image reconstruction after acquiring the complex optical field 
% without using a reference beam.

clear,clc;close all;
%% Parameter
N = 32*32;
gamma = 144;
M = N * gamma; 
gamma_kexi = 4;
kexi = gamma / gamma_kexi;
M_kexi = N * gamma_kexi;

%% Theoreticaltransmission
Tm = raylrnd(sqrt(1/2/M),M,N) .* exp(1i * random('unif',-pi,+pi-eps('double'),M,N)); 
H = hadamard(N);  
Eout1d_had = Tm * H;
dir_curr = pwd;
image_path= fullfile(dir_curr,'data\');
image_file = dir([image_path,'*.png']);
image_file = {image_file.name};
num_file = length(image_file);
image = zeros(sqrt(N),sqrt(N),num_file);
Ein1d = zeros(N,num_file);
for i = 1:num_file
    image(:,:,i) = double(im2gray(imread(fullfile(image_path,image_file{i}))));
    image(:,:,i) = 2*pi*(image(:,:,i) / 255);
    Ein1d(:,i) = reshape(exp(1i*image(:,:,i)),[],1);
end
Eout1d_image = Tm * Ein1d;

%% Measurement
SNR = 20;
Eout1d_had_mea = AddNoise(Eout1d_had,SNR);        
Eout1d_image_mea = AddNoise(Eout1d_image,SNR);  

%% Compressive transmission matrix evaluation
K = Resize_Analytical_bicubic_Antialiasingon(sqrt(M),sqrt(M),sqrt(M_kexi),sqrt(M_kexi));
Tm_mea = Eout1d_had_mea / H;
Ctm_mea = K * Tm_mea;
Eout1d_image_mea_k = K * Eout1d_image_mea;

%% Compressive transmission matrix inversion
[U,S,V] = svd(Ctm_mea);
S_diag = diag(S);
idx = find(S_diag >= max(S_diag)*0.05, 1, 'last');
if isempty(idx)
    idx = 1;
end
S_diag_filter = diag(S_diag(1:idx));
Ctmf_mea = U(:,1:idx) * S_diag_filter * V(:,1:idx)';
Ctmf_mea_inv = pinv(Ctmf_mea);

%% Image reconstruction
Ere1d_k = Ctmf_mea_inv * Eout1d_image_mea_k;
PCC = zeros(num_file,1);
SSIM = zeros(num_file,1);
for i = 1:num_file
    phase_re1d_k = mod(angle(Ere1d_k(:,i)),2*pi);
    PCC(i,1) = corr(phase_re1d_k,reshape(image(:,:,i),[],1));
    SSIM(i,1) = ssim(reshape(phase_re1d_k,sqrt(N),[]),image(:,:,i));
    figure,
    subplot(121);imagesc(image(:,:,i)),title('Ground truth (PCC/SSIM)');axis off;axis image;
    subplot(122);imagesc(reshape(phase_re1d_k,sqrt(N),[])),title(['Reconstruction(',num2str(PCC(i,1),'%.3f'),'/',num2str(SSIM(i,1),'%.3f'),')']);axis off;axis image;
end

%% function
function K = Resize_Analytical_bicubic_Antialiasingon(M_in, N_in, M_out, N_out)
    Kv = get_weights_loop_exact(M_in, M_out);
    Kh = get_weights_loop_exact(N_in, N_out);
    K = kron(Kh, Kv);
end
function W = get_weights_loop_exact(Len_in, Len_out)
    scale = Len_out / Len_in;
    kernel_width = 4.0 / scale;
    A = scale; 
    P = ceil(kernel_width) + 2;
    est_nnz = Len_out * P;
    rows = zeros(est_nnz, 1);
    cols = zeros(est_nnz, 1);
    vals = zeros(est_nnz, 1);
    count = 0;

    for k = 1:Len_out
        u = k / scale + 0.5 * (1 - 1/scale);
        left = floor(u - kernel_width / 2);
        idx_vec = zeros(P, 1);
        w_vec = zeros(P, 1);
        for i = 0 : P-1
            idx = left + i;
            dist = u - idx;
            w = bicubic_kernel_scalar(dist * A);
            idx_vec(i+1) = idx;
            w_vec(i+1) = w;
        end
        w_sum = sum(w_vec);
        if w_sum ~= 0
            w_vec = w_vec / w_sum;
        else
            w_vec(:) = 1/P; 
        end
        for i = 1 : P
            w = w_vec(i);
            if abs(w) < 1e-15; continue; end 
            raw_idx = idx_vec(i);
            aux = raw_idx - 1;
            mapped_idx = raw_idx;
            while mapped_idx < 1 || mapped_idx > Len_in
                 if mapped_idx < 1
                     mapped_idx = 1 - mapped_idx;   
                 elseif mapped_idx > Len_in
                     mapped_idx = 2*Len_in - mapped_idx + 1;
                 end
            end
            
            count = count + 1;
            rows(count) = k;
            cols(count) = mapped_idx;
            vals(count) = w;
        end
    end
    W = sparse(rows(1:count), cols(1:count), vals(1:count), Len_out, Len_in);
end
function w = bicubic_kernel_scalar(x)
    x = abs(x);
    if x <= 1
        w = 1.5 * x^3 - 2.5 * x^2 + 1;
    elseif x < 2
        w = -0.5 * x^3 + 2.5 * x^2 - 4 * x + 2;
    else
        w = 0;
    end
end
function E_n = AddNoise(E,snr)                                               
    [row,col] = size(E);
    signal_power_perColumn = mean(abs(E).^2, 1);    
    noise_power_perColumn = signal_power_perColumn / snr;               
    noise_std_perColumn = sqrt(noise_power_perColumn / 2);                 
    noise_matrix = (randn(row, col) + 1i * randn(row, col));            
    E_n = E + noise_matrix .* noise_std_perColumn; 
end