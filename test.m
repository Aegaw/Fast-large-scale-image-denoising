clear;

a_orig_f0 = single(imread('test_noisy.PNG'));
a_orig_f = single(imread('test_gt.PNG'));

tic;
a_de_3 = adapt_guided(a_orig_f0);
process_time = toc;

fprintf("Original PSNR: %.4f, Denoised PSNR: %.4f, SSIM: %.4f, total time: %.4f second.\n", psnr(a_orig_f0, a_orig_f, 255), psnr(a_de_3, a_orig_f, 255), ssim(a_de_3, a_orig_f), process_time);

figure(1); imshow(a_orig_f0/255); title('Noisy');
figure(2); imshow(a_de_3/255); title('Denoised');
figure(3); imshow(a_orig_f/255); title('Ground');