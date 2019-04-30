close all 
clear all 
Nrea=10^5;
n=1:1:Nrea;
Napp=2*10^4;
Naff=250;
M=5;
a0=1;
a1=1;
P=5;
pas=0.04;
N=10;
%d(n)
w0=0.05*pi;
fi=rand(1,Nrea)*2*pi;
dn(1,:)=sin(n*w0+fi);
%h 
h=[1; 0.5; 0.2; 0.15; 0.3];
%D(n)
for k=5:Nrea
        Dn(:,k)=dn(k:-1:k-M+1);
end
%x(n)
for n=1:Nrea;
    xn(:,n)=a0*h'*Dn(:,n)+a1*(h'*Dn(:,n))^3;
end
%remplir le vecteur X(n)
for k=5:Nrea
        Xn(:,k)=xn(k:-1:k-P+1);
end
%clacul du sorties de neurones
%initialisation des coeff
alpha=rand(N,1);
W=rand(P,N);
s=zeros(1,N);
U=zeros(1,Nrea);
for n=1:Nrea;
    for i=1:N %boucle sur les noeuds
        Ui=W(:,i)'*Xn(:,n);
        s(i)=tanh(Ui); %sortie du noeud i
        U(n)=U(n)+alpha(i)*s(i);
        y(n)=tanh(U(n)); %sortie MLP
        %phase d'apprentissage
        e(n)=dn(n)-y(n);
        if n<=Napp 
             %erreur du controle
            %MAJ des coeff
            W(:,i)=W(:,i)+2*pas*e(n)*alpha(i)*(1-(tanh(Ui))^2)*(1-(tanh(U(n)))^2)*Xn(:,n);
            alpha(i)=alpha(i)+2*pas*e(n)*s(i)*(1-(tanh(U(n)))^2);
        end
    end
    EQM_vec(n)=mean(e.^2);
    
end
%affichage
hold on ; 
plot(dn(1:Naff),'r');
plot(y(1:Naff),'b');


