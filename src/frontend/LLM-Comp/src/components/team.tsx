import React from "react";

import aldi from "../assets/aldi.png";
import pranav from "../assets/pranav.png";
import ryan from "../assets/ryan.png";
import thomas from "../assets/thomas.png";

type Member = {
  name: string;
  role: string;
  img: string;
};

const members: Member[] = [
  {
    name: "Ryan Tran",
    role: "Developer / Documentation",
    img: ryan,
  },
  {
    name: "Aldi Susanto",
    role: "Developer / Documentation / Team Lead",
    img: aldi,
  },
  {
    name: "Thomas Graddy",
    role: "Developer / Documentation",
    img: thomas,
  },
  {
    name: "Pranav Kartha",
    role: "Developer / Documentation",
    img: pranav,
  },
];

export default function TeamSection() {
  return (
    <div style={styles.wrapper}>
      {members.map((m) => (
        <div key={m.name} style={styles.card}>
          <img src={m.img} alt={m.name} style={styles.img} />
          <h3 style={styles.name}>{m.name}</h3>
          <p style={styles.role}>{m.role}</p>
        </div>
      ))}
    </div>
  );
}

const styles = {
  wrapper: {
    display: "flex",
    justifyContent: "center",
    gap: "20px",
    flexWrap: "wrap" as const,
    padding: "20px",
  },
  card: {
    border: "1px solid #555",
    padding: "15px",
    width: "125px",
    textAlign: "center" as const,
    backgroundColor: "#1b1b1b",
  },
  img: {
    width: "100%",
    height: "auto",
    marginBottom: "10px",
  },
  name: {
    color: "white",
    marginBottom: "5px",
  },
  role: {
    color: "#fff3b0ff",
    fontSize: "0.9rem",
  },
};
