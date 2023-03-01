use model::dto::function::ChosenBid;
use model::view::auction::BidProposal;

pub struct Auction;

impl Auction {
    pub fn new() -> Self { Self {} }
}

impl Auction {
    pub fn auction(&self, bids: &[BidProposal]) -> Option<ChosenBid> {
        let mut bids = bids.iter().collect::<Vec<_>>();
        bids.sort_unstable_by(|a, b| a.bid.partial_cmp(&b.bid).unwrap()); // Sort asc
        let first = bids.get(0).cloned().cloned();
        let second = bids.get(1);
        match (first, second) {
            (Some(first), Some(second)) => {
                Some(ChosenBid { price: second.bid, bid: first })
            }
            (Some(first), None) => {
                Some(ChosenBid { price: first.bid, bid: first })
            }
            _ => None,
        }
    }
}
