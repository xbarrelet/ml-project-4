-- En excluant les commandes annulées, quelles sont les commandes récentes de moins de 3 mois que les clients ont reçues avec au moins 3 jours de retard ?
select order_id, order_purchase_timestamp, order_delivered_customer_date, order_estimated_delivery_date
from orders
where order_status != 'canceled'
and order_purchase_timestamp > date('now','-3 months')
and order_delivered_customer_date > date(order_estimated_delivery_date, '+3 days');


-- Qui sont les vendeurs ayant généré un chiffre d'affaires de plus de 100 000 Real sur des commandes livrées via Olist ?
select seller_id, sum(price) as global_amount
from order_items
group by seller_id
having sum(price) > 100000;


-- Qui sont les nouveaux vendeurs (moins de 3 mois d'ancienneté) qui sont déjà très engagés avec la plateforme (ayant déjà vendu plus de 30 produits) ?
select seller_id, count(*) as sold_orders_number, min(shipping_limit_date) as earliest_shipping_date
from order_items oi
group by seller_id
having count(*) > 30
and shipping_limit_date > date('now','-3 months');


-- Quels sont les 5 codes postaux, enregistrant plus de 30 reviews, avec le pire review score moyen sur les 12 derniers mois ?
select s.seller_zip_code_prefix, count(*) as reviews_number, AVG(review_score) as average_review_score
from order_reviews or1
inner join order_items oi on or1.order_id = oi.order_id
inner join sellers s on oi.seller_id = s.seller_id
group by s.seller_zip_code_prefix
having count(*) > 30
and or1.review_creation_date > date('now','-12 months')
order by AVG(review_score) asc
limit 5;