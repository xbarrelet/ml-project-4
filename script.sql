-- En excluant les commandes annulées, quelles sont les commandes récentes de moins de 3 mois que les clients ont reçues avec au moins 3 jours de retard ?

select order_id, order_purchase_timestamp, order_delivered_customer_date, order_estimated_delivery_date
from orders
where order_purchase_timestamp > date('now','-3 months')
and order_delivered_customer_date > date(order_estimated_delivery_date, '+3 days');


-- Qui sont les vendeurs ayant généré un chiffre d'affaires de plus de 100 000 Real sur des commandes livrées via Olist ?

select seller_id, sum(price)
from order_items
group by seller_id
having sum(price) > 100000;


-- Qui sont les nouveaux vendeurs (moins de 3 mois d'ancienneté) qui sont déjà très engagés avec la plateforme (ayant déjà vendu plus de 30 produits) ?



-- Question : Quels sont les 5 codes postaux, enregistrant plus de 30 reviews, avec le pire review score moyen sur les 12 derniers mois ?

