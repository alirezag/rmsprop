data = importdata('testdata.log')
loglog(data.data(:,1:3),'linewidth',1)
set(gca,'fontsize',14); 
xlabel('epoch')
ylabel('cross-entropy loss')
title('performance on MNIST')
legend('SGD','RMSPROP','centered RMSPROP')
print(gcf,'-dpng','sample.png')